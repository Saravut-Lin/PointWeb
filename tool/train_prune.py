import os
import matplotlib.pyplot as plt
import time
import random
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
# Add advanced LR schedulers
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR

from util import dataset, transform, config
from util.s3dis import S3DIS
from util.scannet import ScanNet
from util.market import MarketJamSegDataset # Add Market77 Dataset
from util.util import AverageMeter, intersectionAndUnionGPU


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/s3dis/s3dis_pointweb.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_pointweb.yaml for all options', default=None, nargs=argparse.REMAINDER)
    # Pruning arguments
    parser.add_argument('--prune-method', type=str, choices=['unstructured','structured'], default=None,
                        help='Pruning method to apply during training')
    parser.add_argument('--prune-amount', type=float, default=0.0,
                        help='Fraction of weights to prune each interval')
    parser.add_argument('--prune-interval', type=int, default=1,
                        help='Interval (in epochs) at which to apply pruning')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    # Add pruning arguments to cfg if not present
    if not hasattr(cfg, 'prune_method'):
        cfg.prune_method = args.prune_method
    if not hasattr(cfg, 'prune_amount'):
        cfg.prune_amount = args.prune_amount
    if not hasattr(cfg, 'prune_interval'):
        cfg.prune_interval = args.prune_interval
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    # also log to file in the save_path directory
    if hasattr(args, 'save_path'):
        log_file = os.path.join(args.save_path, 'train.log')
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def init():
    global args, logger, writer
    args = get_parser()
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
    logger.info(args)


def main():
    init()
    # lists to store metrics per epoch
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    # Model selection
    if args.arch == 'pointnet_seg':
        from model.pointnet.pointnet import PointNetSeg as Model
    elif args.arch == 'pointnet2_seg':
        from model.pointnet2.pointnet2_seg import PointNet2SSGSeg as Model
    elif args.arch == 'pointweb_seg':
        from model.pointweb.pointweb_seg import PointWebSeg as Model
    else:
        raise Exception(f'architecture not supported: {args.arch}')

    model = Model(c=args.fea_dim, k=args.classes, use_xyz=args.use_xyz)
    if args.sync_bn:
        from util.util import convert_to_syncbn
        convert_to_syncbn(model)
    model = torch.nn.DataParallel(model.cuda())

    # Initial pruning if requested
    if args.prune_method and args.prune_amount > 0.0:
        logger.info(f"Initial pruning: method={args.prune_method}, amount={args.prune_amount}")
        if args.prune_method == 'unstructured':
            model.module.apply_unstructured_pruning(amount=args.prune_amount)
        else:
            model.module.apply_structured_pruning(amount=args.prune_amount)

    # Criterion with optional class weights
    weight = None
    if isinstance(args.weight, (list, tuple)):
        weight = torch.tensor(args.weight, device='cuda')
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=args.ignore_label).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_epoch, gamma=args.multiplier)
    # Option A: Cosine Annealing with Warm Restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.step_epoch,    # first cycle length (in epochs)
        T_mult=2,               # cycle length multiplier
        eta_min=1e-5            # minimum learning rate
    )
    # Option B (alternative): Cyclical LR (uncomment if preferred)
    # scheduler = CyclicLR(
    #     optimizer,
    #     base_lr=1e-5,
    #     max_lr=args.base_lr,
    #     step_size_up=len(train_loader)*args.step_epoch,
    #     mode='triangular2',
    #     cycle_momentum=False
    # )

    # Load pretrained checkpoint if provided as string
    if isinstance(args.weight, str) and args.weight:
        if os.path.isfile(args.weight):
            logger.info(f"=> loading checkpoint '{args.weight}'")
            ckpt = torch.load(args.weight, map_location='cuda')
            model.load_state_dict(ckpt['state_dict'])
            logger.info(f"=> loaded checkpoint '{args.weight}'")
        else:
            logger.warning(f"No checkpoint found at '{args.weight}', training from scratch.")

    # Resume training state
    if args.resume and os.path.isfile(args.resume):
        logger.info(f"=> loading resume checkpoint '{args.resume}'")
        ckpt = torch.load(args.resume, map_location='cuda')
        args.start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        logger.info(f"=> resumed from '{args.resume}' (epoch {args.start_epoch})")

    # Data transforms and datasets
    train_transform = transform.Compose([transform.ToTensor()])
    if args.data_name == 'market':
        train_data = MarketJamSegDataset(h5_path=args.data_root,
                                         split='train',
                                         transform=None)
    elif args.data_name == 's3dis':
        train_data = S3DIS(split='train', data_root=args.train_full_folder,
                           num_point=args.num_point, test_area=args.test_area,
                           block_size=args.block_size, sample_rate=args.sample_rate,
                           transform=train_transform)
    elif args.data_name == 'scannet':
        train_data = ScanNet(split='train', data_root=args.data_root,
                              num_point=args.num_point, block_size=args.block_size,
                              sample_rate=args.sample_rate, transform=train_transform)
    else:
        raise Exception(f"Unsupported data_name: {args.data_name}")

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.train_batch_size,
                                               shuffle=True,
                                               num_workers=args.train_workers,
                                               pin_memory=True,
                                               worker_init_fn=worker_init_fn)

    val_loader = None
    if args.evaluate:
        val_transform = transform.Compose([transform.ToTensor()])
        if args.data_name == 'market':
            val_data = MarketJamSegDataset(h5_path=args.data_root,
                                           split='test', transform=val_transform)
        else:
            # fallback
            val_data = dataset.PointData(split='val', data_root=args.data_root,
                                         data_list=args.val_list, transform=val_transform)
        val_loader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=args.train_batch_size_val,
                                                 shuffle=False,
                                                 num_workers=args.train_workers,
                                                 pin_memory=True)

    # Early stopping & warm-up
    best_mIoU = 0.0
    no_improve = 0
    best_epoch = 0
    warmup_epochs = getattr(args, 'early_stop_warmup', 20)

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()

        # Periodic pruning
        if args.prune_method and args.prune_amount > 0.0 and (epoch+1) % args.prune_interval == 0:
            logger.info(f"Applying pruning at epoch {epoch+1}: method={args.prune_method}, amount={args.prune_amount}")
            if args.prune_method == 'unstructured':
                model.module.apply_unstructured_pruning(amount=args.prune_amount)
            else:
                model.module.apply_structured_pruning(amount=args.prune_amount)

        # Train + log
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(
            train_loader, model, criterion, optimizer, epoch)
        writer.add_scalar('loss_train', loss_train, epoch+1)
        writer.add_scalar('mIoU_train', mIoU_train, epoch+1)

        # record training metrics
        train_loss_list.append(loss_train)
        train_acc_list.append(allAcc_train)


        # --- SAVE EVERY EPOCH AFTER WARM-UP ---
        if epoch + 1 > warmup_epochs:
            ckpt_path = os.path.join(
                args.save_path, f'train_epoch_{epoch+1}.pth')
            logger.info(f"Saving checkpoint (post-warmup): {ckpt_path}")
            torch.save({
                'epoch':      epoch+1,
                'state_dict': model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'scheduler':  scheduler.state_dict(),
            }, ckpt_path)
        else:
            logger.info(f"[Warm-up] Epoch {epoch+1}/{warmup_epochs} (no save)")

        # Validate + early-stop
        if args.evaluate and val_loader is not None:
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(
                val_loader, model, criterion)
            writer.add_scalar('loss_val', loss_val, epoch+1)
            writer.add_scalar('mIoU_val', mIoU_val, epoch+1)

            # record validation metrics
            val_loss_list.append(loss_val)
            val_acc_list.append(allAcc_val)

            if epoch + 1 > warmup_epochs:
                if mIoU_val > best_mIoU:
                    best_mIoU = mIoU_val
                    best_epoch = epoch+1
                    no_improve = 0
                    best_path = os.path.join(args.save_path, 'best_model.pth')
                    torch.save(model.state_dict(), best_path)
                    logger.info(f"New best mIoU: {best_mIoU:.4f}, saved to {best_path}")
                else:
                    no_improve += 1
                    logger.info(f"No improvement for {no_improve} epochs post-warmup")
                if no_improve >= args.early_stop_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                # still warming up
                no_improve = 0
                logger.info(f"[Warm-up] Skipping early-stop check at epoch {epoch+1}")

    logger.info(f"Training complete. Best validation mIoU {best_mIoU:.4f} achieved at epoch {best_epoch}.")

    # Remove pruning reparameterizations to finalize weights
    if args.prune_method and args.prune_amount > 0.0:
        logger.info("Removing pruning masks and finalizing weights")
        model.module.remove_pruning()

    writer.close()

    # plot and save loss & accuracy curves
    epochs = list(range(1, len(train_loss_list) + 1))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    ax1.plot(epochs, train_loss_list, label='Train Loss')
    ax1.plot(epochs, val_loss_list, label='Val Loss')
    ax1.set_title('Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax2.plot(epochs, train_acc_list, label='Train Acc')
    ax2.plot(epochs, val_acc_list, label='Val Acc')
    ax2.set_title('Accuracy per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    metrics_path = os.path.join(args.save_path, 'metrics.png')
    plt.tight_layout()
    plt.savefig(metrics_path)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(input)
        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0:
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
        writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
        writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
        writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls
        output = model(input)
        loss = criterion(output, target)

        output = output.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0:
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(args.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    main()



"""
sh tool/train_prune.sh market pointweb --prune-method unstructured --prune-amount 0.2 --prune-interval 2
"""