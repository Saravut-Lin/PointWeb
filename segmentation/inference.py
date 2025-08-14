#!/usr/bin/env python3
"""
segmentation_inference.py: Unified script for running PointWeb segmentation inference
with streaming HDF5 loading, hard-coded dataset paths, GPU/CUDA usage,
and saving three masks per sample: color_mask, gt_mask, pred_mask.
Also handles real-world PCD segmentation with NaN removal and optional HDF5 sample visualization.
"""
import os
import sys
import logging
from pathlib import Path

import torch
import numpy as np
import h5py
import random
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm

# === User-configurable paths and parameters ===
CHECKPOINT_PATH = "/home/s2671222/pointweb_gcp/exp/market/pointweb/optim_distil/model/best_model.pth"
HDF5_INPUT_PATH  = "/home/s2671222/pointweb_gcp/dataset/market77/jam_hartleys_strawberry_300gm_1200_2048_segmentation_20480_12000"
HDF5_OUTPUT_PATH = "/home/s2671222/pointweb_gcp/segmentation/results/masks_optim_distil.h5"
PCD_INPUT_PATH   = "/home/s2671222/pointweb_gcp/segmentation/realworld_scene/realworld_scene_1.pcd"
PCD_OUTPUT_PATH  = "/home/s2671222/pointweb_gcp/segmentation/results/scene_segmented1_optim_distil.ply"

# Inference parameters
BATCH_SIZE = 2              # smaller to fit GPU memory
NUM_CLASSES = 2
EXTRA_FEAT_CHANNELS = 3     # features beyond x,y,z (e.g., RGB + padding)
USE_XYZ = True
NUM_POINTS_PCD = 20480
GRID_SIZE    = 0.389 #0.2      # chunk size in meters
OVERLAP_RATIO = 0.3     # overlap fraction between chunks

# Visualization: number of HDF5 samples to output PLYs (0 to skip)
VISUALIZE_SAMPLES = 10
VIS_SAVE_DIR = "/home/s2671222/pointweb_gcp/segmentation/vis_samples_optim_distil"

# === Setup import paths ===
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))  # for model package
sys.path.insert(0, str(HERE))       # for local modules
from item_pointweb_torch import PointWebSegHead

# CUDA device
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please install GPU drivers / CUDA toolkit.")
device = torch.device('cuda')


def strip_prefix_from_state_dict(state_dict, prefix='module.'):
    new = {}
    for k, v in state_dict.items():
        new_key = k[len(prefix):] if k.startswith(prefix) else k
        new[new_key] = v
    return new


def load_model(ckpt_path, num_classes, extra_feat, use_xyz, device):
    model = PointWebSegHead(c=extra_feat, k=num_classes, use_xyz=use_xyz, device=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = strip_prefix_from_state_dict(ckpt.get('state_dict', ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    logging.info(f"Loaded PointWebSegHead: {ckpt_path}")
    return model


def infer_h5():
    """Stream HDF5 samples in batches and predict segmentation."""
    model = load_model(CHECKPOINT_PATH, NUM_CLASSES, EXTRA_FEAT_CHANNELS, USE_XYZ, device)

    with h5py.File(HDF5_INPUT_PATH, 'r') as fin, \
         h5py.File(HDF5_OUTPUT_PATH, 'w', libver='latest') as fout:
        pts_ds  = fin['seg_points']   # [N, P, 3]
        cols_ds = fin['seg_colors']   # [N, P, 3]
        lbl_ds  = fin['seg_labels']   # [N, P]
        N, P, _ = pts_ds.shape
        logging.info(f"HDF5: {N} samples, {P} points each")

        # Virtual datasets for color and GT masks
        vsrc_color = h5py.VirtualSource(HDF5_INPUT_PATH, 'seg_colors', shape=(N, P, 3))
        layout_color = h5py.VirtualLayout(shape=(N, P, 3), dtype=cols_ds.dtype)
        layout_color[...] = vsrc_color; fout.create_virtual_dataset('color_mask', layout_color)

        vsrc_gt = h5py.VirtualSource(HDF5_INPUT_PATH, 'seg_labels', shape=(N, P))
        layout_gt = h5py.VirtualLayout(shape=(N, P), dtype=lbl_ds.dtype)
        layout_gt[...] = vsrc_gt; fout.create_virtual_dataset('gt_mask', layout_gt)

        # Dataset to store predictions
        pred_ds = fout.create_dataset('pred_mask', shape=(N, P), dtype='uint8')

        # Batch-wise inference
        for start in tqdm(range(0, N, BATCH_SIZE), desc='Inferring HDF5'):
            end = min(start + BATCH_SIZE, N)
            pts = pts_ds[start:end]; cols = cols_ds[start:end]

            # Build features: xyz + padded extras
            extras = cols
            pad_dim = EXTRA_FEAT_CHANNELS - extras.shape[2]
            if pad_dim>0:
                extras = np.pad(extras, ((0,0),(0,0),(0,pad_dim)), mode='constant')
            feats = np.concatenate([pts, extras], axis=2)  # [B, P, 3+EX]

            x = torch.from_numpy(feats).float().to(device)
            inp = (x.transpose(2,1).contiguous() if USE_XYZ else x[...,3:].transpose(2,1).contiguous())
            with torch.no_grad(): logits = model(inp)
            preds = logits.argmax(1).cpu().numpy().astype('uint8')
            pred_ds[start:end] = preds

    logging.info(f"Saved predictions to {HDF5_OUTPUT_PATH}")


# --- Visualization helper ---
# After running inference, launch:
#     visualize_h5_sample(sample_idx=0, max_points=5000)
# to display the color cloud, GT, and prediction masks for one sample.
def visualize_h5_sample(sample_idx=0, max_points=None, out_dir=VIS_SAVE_DIR):
    """Save color, GT, and predicted segmentation as PLYs and a combined figure."""
    import h5py
    # Load data
    with h5py.File(HDF5_INPUT_PATH, 'r') as fin:
        pts   = fin['seg_points'][sample_idx]
        cols  = fin['seg_colors'][sample_idx]
        gt_raw = fin['seg_labels'][sample_idx]  # may be one-hot or shape [P,2]
        # Convert to 1D integer labels if needed
        if gt_raw.ndim == 2 and gt_raw.shape[1] == NUM_CLASSES:
            gt = gt_raw.argmax(axis=1)
        else:
            gt = gt_raw.squeeze()
    with h5py.File(HDF5_OUTPUT_PATH, 'r') as fout:
        pred  = fout['pred_mask'][sample_idx]
    # Optionally downsample; if max_points is None, show all
    if max_points is None:
        idxs = np.arange(pts.shape[0])
    else:
        N = pts.shape[0]
        idxs = np.random.choice(N, max_points, replace=False) if N > max_points else np.arange(N)

    # Prepare output directory
    os.makedirs(out_dir, exist_ok=True)

    # 1) Save PLYs
    # Color cloud
    p_c = o3d.geometry.PointCloud()
    p_c.points = o3d.utility.Vector3dVector(pts[idxs])
    p_c.colors = o3d.utility.Vector3dVector(cols[idxs])
    o3d.io.write_point_cloud(os.path.join(out_dir, f"sample_{sample_idx:03d}_color.ply"), p_c)
    # GT mask
    cmap = plt.get_cmap('tab10')
    p_gt = o3d.geometry.PointCloud()
    p_gt.points = o3d.utility.Vector3dVector(pts[idxs])
    gt_colors = np.ascontiguousarray(cmap(gt[idxs] % NUM_CLASSES)[:,:3], dtype=np.float32)
    p_gt.colors = o3d.utility.Vector3dVector(gt_colors)
    o3d.io.write_point_cloud(os.path.join(out_dir, f"sample_{sample_idx:03d}_gt.ply"), p_gt)
    # Pred mask
    p_pr = o3d.geometry.PointCloud()
    p_pr.points = o3d.utility.Vector3dVector(pts[idxs])
    pr_colors = np.ascontiguousarray(cmap(pred[idxs] % NUM_CLASSES)[:,:3], dtype=np.float32)
    p_pr.colors = o3d.utility.Vector3dVector(pr_colors)
    o3d.io.write_point_cloud(os.path.join(out_dir, f"sample_{sample_idx:03d}_pred.ply"), p_pr)

    # 2) Save combined figure
    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot(1,3,1, projection='3d')
    ax.scatter(pts[idxs,0], pts[idxs,1], pts[idxs,2], c=cols[idxs], s=1)
    ax.set_title('Color Cloud')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax = fig.add_subplot(1,3,2, projection='3d')
    ax.scatter(pts[idxs,0], pts[idxs,1], pts[idxs,2], c=cmap(gt[idxs] % NUM_CLASSES), s=1)
    ax.set_title('GT Segmentation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax = fig.add_subplot(1,3,3, projection='3d')
    ax.scatter(pts[idxs,0], pts[idxs,1], pts[idxs,2], c=cmap(pred[idxs] % NUM_CLASSES), s=1)
    ax.set_title('Pred Segmentation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig_path = os.path.join(out_dir, f"sample_{sample_idx:03d}_comparison.png")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)

    logging.info(f"Saved PLYs and figure for sample {sample_idx} in {out_dir}")


def infer_pcd():
    """Segment a full PCD by chunking into NUM_POINTS_PCD windows."""
    model = load_model(CHECKPOINT_PATH, NUM_CLASSES, EXTRA_FEAT_CHANNELS, USE_XYZ, device)
    pcd = o3d.io.read_point_cloud(PCD_INPUT_PATH)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(pts)
    mask = ~np.isnan(pts).any(1)
    pts, cols = pts[mask], cols[mask]
    n = len(pts)
    logging.info(f"PCD loaded: {n} points post-NaN removal")

    full_colors = np.zeros((n,3),dtype=np.float32)
    cmap = plt.get_cmap('tab10')
    
    '''
    # loop over chunks
    for start in tqdm(range(0,n,NUM_POINTS_PCD),desc='Chunking PCD'):
        end = min(start+NUM_POINTS_PCD,n)
        idxs = np.arange(start,end)
        chunk_pts = pts[idxs]
        chunk_cols = cols[idxs]
        # pad if last chunk smaller
        if len(idxs)<NUM_POINTS_PCD:
            pad_n = NUM_POINTS_PCD - len(idxs)
            chunk_pts = np.vstack([chunk_pts,np.zeros((pad_n,3))])
            chunk_cols = np.vstack([chunk_cols,np.zeros((pad_n,3))])
        # features
        extras = chunk_cols
        pad = EXTRA_FEAT_CHANNELS - extras.shape[1]
        if pad>0:
            extras = np.pad(extras,((0,0),(0,pad)),mode='constant')
        feats = np.concatenate([chunk_pts,extras],axis=1)[None]  # [1,P,3+E]
        x = torch.from_numpy(feats).float().to(device)
        inp = x.transpose(2,1).contiguous() if USE_XYZ else x[...,3:].transpose(2,1).contiguous()
        with torch.no_grad(): logits = model(inp)
        pred = logits.argmax(1).cpu().numpy().squeeze()[:len(idxs)]
        full_colors[idxs] = cmap(pred % NUM_CLASSES)[:,:3]
    '''
    #------------------
        # --- Spatial partitioning by voxel grid ---
    min_bound = pts.min(axis=0)
    max_bound = pts.max(axis=0)
    step = GRID_SIZE * (1 - OVERLAP_RATIO)
    x_starts = np.arange(min_bound[0], max_bound[0], step)
    y_starts = np.arange(min_bound[1], max_bound[1], step)

    # accumulator for votes
    vote_counts = np.zeros((n, NUM_CLASSES), dtype=np.int32)

    for x0 in tqdm(x_starts, desc='Chunk X'):
        for y0 in y_starts:
            mask = ((pts[:,0] >= x0) & (pts[:,0] < x0 + GRID_SIZE) &
                    (pts[:,1] >= y0) & (pts[:,1] < y0 + GRID_SIZE))
            idxs = np.nonzero(mask)[0]
            if idxs.size == 0:
                continue
            # sample or pad to NUM_POINTS_PCD
            if idxs.shape[0] >= NUM_POINTS_PCD:
                sel = np.random.choice(idxs, NUM_POINTS_PCD, replace=False)
            else:
                pad = NUM_POINTS_PCD - idxs.shape[0]
                sel = np.concatenate([idxs, np.full(pad, idxs[0], dtype=int)])
            # build features
            chunk_pts, chunk_cols = pts[sel], cols[sel]
            extras = chunk_cols
            pad_dim = EXTRA_FEAT_CHANNELS - extras.shape[1]
            if pad_dim > 0:
                extras = np.pad(extras, ((0,0),(0,pad_dim)), mode='constant')
            feats = np.concatenate([chunk_pts, extras], axis=1)[None]
            x_in = torch.from_numpy(feats).float().to(device)
            inp = x_in.transpose(2,1).contiguous() if USE_XYZ else x_in[...,3:].transpose(2,1).contiguous()
            with torch.no_grad():
                logits = model(inp)
            pred = logits.argmax(1).cpu().numpy().squeeze()[:len(sel)]
            # tally votes
            for local_i, global_i in enumerate(sel):
                vote_counts[global_i, pred[local_i]] += 1

    # finalize by majority vote
    final_pred = vote_counts.argmax(axis=1)
    full_colors = cmap(final_pred % NUM_CLASSES)[:,:3]
        #--------------

    # --- Save original color cloud ---
    orig_pcd = o3d.geometry.PointCloud()
    orig_pcd.points = o3d.utility.Vector3dVector(pts)
    orig_pcd.colors = o3d.utility.Vector3dVector(cols)
    orig_path = PCD_OUTPUT_PATH.replace('.ply', '_color.ply')
    o3d.io.write_point_cloud(orig_path, orig_pcd)

    # write output
    seg = o3d.geometry.PointCloud()
    seg.points = o3d.utility.Vector3dVector(pts)
    seg.colors = o3d.utility.Vector3dVector(full_colors)
    os.makedirs(os.path.dirname(PCD_OUTPUT_PATH),exist_ok=True)
    o3d.io.write_point_cloud(PCD_OUTPUT_PATH,seg)
    logging.info(f"Saved segmented PCD to {PCD_OUTPUT_PATH}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    infer_h5()
    
    # Optional HDF5 visualization
    if VISUALIZE_SAMPLES > 0:
        os.makedirs(VIS_SAVE_DIR, exist_ok=True)
        # randomly select samples to visualize
        with h5py.File(HDF5_INPUT_PATH, 'r') as fin:
            num_samples = fin['seg_points'].shape[0]
        sample_indices = random.sample(range(num_samples), min(VISUALIZE_SAMPLES, num_samples))
        for idx in sample_indices:
            visualize_h5_sample(sample_idx=idx)


    # Run PCD inference if desired
    #infer_pcd()


if __name__ == '__main__':
    main()
