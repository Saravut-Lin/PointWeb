# PointWeb/util/market.py

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class MarketJamSegDataset(Dataset):
    """
    A PyTorch Dataset that reads exactly one 20 480‐point sample (XYZ+RGB + one‐hot mask)
    from a single HDF5 file on disk, then converts the one-hot mask to integer labels.

    Assumes the HDF5 file has three datasets:
      - "seg_points" : shape = (N, 20480, 3), dtype=float32
      - "seg_colors" : shape = (N, 20480, 3), dtype=float32
      - "seg_labels" : shape = (N, 20480, 2), dtype=float32  (one-hot: [1,0] for background, [0,1] for target)

    We never load all N samples into RAM at once; __getitem__ simply does:
      pts = f["seg_points"][idx]      # (20480,3)
      cols = f["seg_colors"][idx]      # (20480,3)
      lbls_onehot = f["seg_labels"][idx]  # (20480,2)
      lbls_int = argmax(lbls_onehot, axis=1)  → (20480,) in {0,1}

    Finally, we concatenate (pts, cols) → (20480,6) and return (torch.FloatTensor, torch.LongTensor).
    """

    def __init__(self,
                 h5_path: str,
                 split: str = 'train',
                 transform=None):
        """
        Args:
            h5_path (str): full path to the single HDF5 file.
            split (str): either "train" or "val" or "test" (for naming only; we assume
                         the same file contains all splits and you'll index it accordingly).
            transform (callable, optional): if not None, a function that takes
                         (points_rgb: np.ndarray of shape (20480,6),
                          labels_int: np.ndarray of shape (20480,))
                         and returns transformed (points_rgb, labels_int).
        """
        super().__init__()
        self.h5_path = h5_path
        self.split = split
        self.transform = transform

        # Open the HDF5 file but do not load into memory.  We keep it open for indexed access.
        if not os.path.isfile(self.h5_path):
            raise FileNotFoundError(f"Cannot find HDF5 file: {self.h5_path}")
        self.h5_file = h5py.File(self.h5_path, 'r')

        # Datasets inside:
        #    "seg_points" -> shape (N, 20480, 3)
        #    "seg_colors" -> shape (N, 20480, 3)
        #    "seg_labels" -> shape (N, 20480, 2)
        self.points_ds = self.h5_file["seg_points"]
        self.colors_ds = self.h5_file["seg_colors"]
        self.labels_ds = self.h5_file["seg_labels"]



        # Sanity check: they must all have the same first dimension:
        n_pts = self.points_ds.shape[0]
        if self.colors_ds.shape[0] != n_pts or self.labels_ds.shape[0] != n_pts:
            raise RuntimeError(f"HDF5 file {self.h5_path} is inconsistent: "
                               f"seg_points first dim = {n_pts}, "
                               f"seg_colors first dim = {self.colors_ds.shape[0]}, "
                               f"seg_labels first dim = {self.labels_ds.shape[0]}.")


        #self.num_samples = n_pts
        #print(f"[MarketJamSegDataset] Found {self.num_samples} total samples in HDF5 ({self.split}).")
        # Determine split indices: 80% train, 20% val (first 80% for train, last 20% for val)
        num_train = int(0.8 * n_pts)
        if self.split == 'train':
            self.indices = list(range(0, num_train))
        elif self.split == 'test':
            self.indices = list(range(num_train, n_pts))
        else:
            # fallback: use entire range
            self.indices = list(range(n_pts))
        self.num_samples = len(self.indices)
        print(f"[MarketJamSegDataset] Using split '{self.split}' with {self.num_samples} samples.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Map idx to the actual dataset index for the split
        actual_idx = self.indices[idx]
        # 1) Read exactly one sample from HDF5
        #pts_xyz = self.points_ds[idx]    # shape: (20480, 3), dtype=float32
        #cols_rgb = self.colors_ds[idx]   # shape: (20480, 3), dtype=float32
        #onehot = self.labels_ds[idx]     # shape: (20480, 2), dtype=float32
        pts_xyz = self.points_ds[actual_idx]    # shape: (20480, 3), dtype=float32
        cols_rgb = self.colors_ds[actual_idx]   # shape: (20480, 3), dtype=float32
        onehot = self.labels_ds[actual_idx]     # shape: (20480, 2), dtype=float32

        # 2) Combine xyz + rgb → (20480, 6)
        '''
        #    (Optionally, you could normalize XYZ or colors here if desired.)
        points_rgb = np.concatenate((pts_xyz, cols_rgb), axis=1)  # (20480, 6)
        zeros = np.zeros((points_rgb.shape[0], 3), dtype=points_rgb.dtype)
        points_rgb = np.concatenate((points_rgb, zeros), axis=1)  # (20480, 9)
        '''
        # 3) Combine xyz + rgb → (20480, 6)
        points_feats = np.concatenate((pts_xyz, cols_rgb), axis=1)

        '''
        # 3) Read precomputed normals and combine features
        normals = self.normals_ds[actual_idx]  # shape: (20480, 3)
        points_feats = np.concatenate((pts_rgb, normals), axis=1)  # (20480, 9)
        '''

        # 3) Convert one‐hot labels → integer labels {0, 1}
        #    We assume onehot[i] is exactly something like [1,0] or [0,1].
        labels_int = np.argmax(onehot, axis=1).astype(np.int64)   # (20480,)

        # 4) (Optional) Apply any transforms that expect numpy arrays:
        if self.transform is not None:
            #points_rgb, labels_int = self.transform(points_rgb, labels_int)
            points_feats, labels_int = self.transform(points_feats, labels_int)

        # 5) Convert to torch.Tensors (handle both numpy arrays and tensors):
        #if isinstance(points_rgb, np.ndarray):
        #    pts_tensor = torch.from_numpy(points_rgb).float()
        if isinstance(points_feats, np.ndarray):
            pts_tensor = torch.from_numpy(points_feats).float()        
        else:
            #pts_tensor = points_rgb.float()
            pts_tensor = points_feats.float()

        if isinstance(labels_int, np.ndarray):
            lbl_tensor = torch.from_numpy(labels_int).long()
        else:
            lbl_tensor = labels_int.long()


        return pts_tensor, lbl_tensor

    def close(self):
        # Call this if you want to explicitly close the HDF5 file
        self.h5_file.close()