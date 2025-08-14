# test_market.py
import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from util import config
from util.util import AverageMeter, intersectionAndUnion, check_makedirs
import pickle
from util.market import MarketJamSegDataset   

def get_parser():
    parser = argparse.ArgumentParser(description='Test PointWeb (or PointNet, etc.) on Market77 segmentation')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/market/market_pointweb.yaml', 
        help='path to a YAML config that contains:  dataset path, model path, batch size, etc.'
    )
    parser.add_argument(
        'opts',
        help='modify config options from command line',
        nargs=argparse.REMAINDER
    )
    return parser.parse_args()

def get_logger():
    logger = logging.getLogger("market-tester")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s"
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)
    return logger

def main():
    args = get_parser()
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts:
        cfg = config.merge_cfg_from_list(cfg, args.opts)

    logger = get_logger()
    logger.info("Config:\n%s", cfg)

    # 1) build the model exactly as in train.py / test_s3dis.py:
    arch = cfg.arch.lower()
    num_classes = cfg.classes
    feat_dim = cfg.fea_dim
    use_xyz = cfg.use_xyz

    if arch == 'pointnet_seg':
        from model.pointnet.pointnet import PointNetSeg as Model
    elif arch == 'pointnet2_seg':
        from model.pointnet2.pointnet2_seg import PointNet2SSGSeg as Model
    elif arch == 'pointweb_seg':
        from model.pointweb.pointweb_seg import PointWebSeg as Model
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    model = Model(c=feat_dim, k=num_classes, use_xyz=use_xyz)
    model = torch.nn.DataParallel(model.cuda())
    logger.info("Loaded model architecture:\n%s", model)

    # 2) load checkpoint
    if not os.path.isfile(cfg.model_path):
        raise FileNotFoundError(f"Cannot find model checkpoint: {cfg.model_path}")
    ckpt = torch.load(cfg.model_path, map_location='cuda')
    #model.load_state_dict(ckpt['state_dict'], strict=False)
    # support both wrapped checkpoint and raw state_dict
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded checkpoint from %s", cfg.model_path)

    # 3) create dataset + loader
    h5_path = cfg.data_root
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"Cannot find HDF5 file: {h5_path}")

    # We set batch_size=1 because we will manually split each 20 480‐point sample
    test_dataset = MarketJamSegDataset(
        h5_path=h5_path,
        split='test',
        transform=None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,                # one sample (20 480 points) at a time
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    logger.info("Found %d total samples in HDF5 (Market77)", len(test_dataset))
    check_makedirs(cfg.save_folder)
    pred_save, gt_save = [], []

    # 4) start testing
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()

    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    batch_time = AverageMeter()

    with torch.no_grad():
        end_event = torch.cuda.Event(enable_timing=True)
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        for i, (pts_rgb, lbl_int) in enumerate(test_loader):
            # pts_rgb: (1, N, 6), lbl_int: (1, N)
            pts_rgb = pts_rgb.squeeze(0)  # (N, 6)
            lbl_int = lbl_int.squeeze(0)  # (N,)
            N = pts_rgb.shape[0]
            num_point = cfg.num_point
            # We'll accumulate a prediction for each of the N points:
            full_pred = torch.zeros(N, dtype=torch.long, device='cpu')

            # Process in chunks of size num_point:
            for start in range(0, N, num_point):
                end = min(start + num_point, N)
                chunk_pts = pts_rgb[start:end, :]     # (chunk_size, 6)
                chunk_lbl = lbl_int[start:end]        # (chunk_size,)

                chunk_size = end - start
                if chunk_size < num_point:
                    # pad to exactly num_point rows with zeros
                    pad_size = num_point - chunk_size
                    pad_pts = torch.zeros(pad_size, 6, dtype=chunk_pts.dtype, device=chunk_pts.device)
                    pad_lbl = torch.zeros(pad_size, dtype=chunk_lbl.dtype, device=chunk_lbl.device)
                    chunk_pts = torch.cat([chunk_pts, pad_pts], dim=0)
                    chunk_lbl = torch.cat([chunk_lbl, pad_lbl], dim=0)
                    valid_len = chunk_size
                else:
                    valid_len = num_point
                chunk_pts_tensor = chunk_pts.unsqueeze(0).contiguous().cuda()  # shape = (1, num_point, 6)
                output = model(chunk_pts_tensor)
                pred_chunk = output.max(dim=1)[1].squeeze(0)
                full_pred[start:start+valid_len] = pred_chunk[:valid_len].cpu()
                # Clear intermediate CUDA tensors for this chunk
                del chunk_pts_tensor, output, pred_chunk
                torch.cuda.empty_cache()
            gt_numpy = lbl_int.cpu().numpy()
            pred_numpy = full_pred.numpy()
            pred_save.append(pred_numpy)
            gt_save.append(gt_numpy)

            intersection, union, target = intersectionAndUnion(
                pred_numpy, gt_numpy, num_classes, cfg.ignore_label
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)

            # timing:
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event) / 1000.0
            batch_time.update(elapsed)
            start_event.record()

            if ((i + 1) % cfg.print_freq == 0) or (i + 1 == len(test_loader)):
                acc = intersection_meter.val.sum() / (target_meter.val.sum() + 1e-10)
                logger.info(
                    "[%d/%d] Time: %.3f  Acc: %.4f",
                    i+1, len(test_loader), batch_time.val, acc
                )

    # 5) final metrics:
    iou_per_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    acc_per_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_per_class)
    mAcc = np.mean(acc_per_class)
    allAcc = intersection_meter.sum.sum() / (target_meter.sum.sum() + 1e-10)

    logger.info("===================================")
    logger.info("Finished testing on Market77:")
    logger.info("  mIoU   = %.4f", mIoU)
    logger.info("  mAcc   = %.4f", mAcc)
    logger.info("  allAcc = %.4f", allAcc)
    for cls_idx in range(num_classes):
        logger.info("  class %d -> IoU %.4f  Acc %.4f",
                    cls_idx, iou_per_class[cls_idx], acc_per_class[cls_idx])
    logger.info("===================================")

    # Save all predictions and ground truths
    with open(os.path.join(cfg.save_folder, "pred.pickle"), "wb") as f:
        pickle.dump({'pred': pred_save}, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(cfg.save_folder, "gt.pickle"), "wb") as f:
        pickle.dump({'gt': gt_save}, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()