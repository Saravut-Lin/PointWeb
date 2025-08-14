import os
import numpy as np

from torch.utils.data import Dataset


class S3DIS(Dataset):
    def __init__(
        self,
        split='train',
        data_root='trainval_fullarea',
        num_point=4096,
        test_area=5,
        block_size=1.0,
        sample_rate=1.0,
        transform=None
    ):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform

        # 1) Gather all room filenames (no loading yet)
        rooms = sorted(os.listdir(data_root))
        rooms = [r for r in rooms if 'Area_' in r]

        if split == 'train':
            rooms_split = [r for r in rooms if f'Area_{test_area}' not in r]
        else:
            rooms_split = [r for r in rooms if f'Area_{test_area}' in r]

        # 2) We will store, per room:
        #    - the path (so we can np.load(...) in __getitem__)
        #    - its min/max coord (tiny arrays)
        #    - the total number of points (to build sampling probabilities)
        self.room_files = []
        self.room_coord_min = []
        self.room_coord_max = []
        num_point_all = []

        for room_name in rooms_split:
            room_path = os.path.join(data_root, room_name)
            # Load exactly once just to compute bounds and count, then delete:
            room_data = np.load(room_path)       # shape [N, 7]
            points = room_data[:, 0:6]           # [N, 6]
            labels = room_data[:, 6]             # [N]
            coord_min = np.amin(points, axis=0)[:3]  # xyz min
            coord_max = np.amax(points, axis=0)[:3]  # xyz max

            # Save path and bounding‐box info
            self.room_files.append(room_path)
            self.room_coord_min.append(coord_min)
            self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)

            # Immediately free that array to avoid keeping it in RAM
            del room_data
            del points
            del labels

        # 3) Build sampling probabilities exactly as before
        num_point_all = np.array(num_point_all, dtype=np.float64)
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)

        room_idxs = []
        for idx_room, prob in enumerate(sample_prob):
            # Round to integer number of samples for this room
            count = int(round(prob * num_iter))
            room_idxs.extend([idx_room] * count)

        self.room_idxs = np.array(room_idxs, dtype=np.int64)
        print(f"Totally {len(self.room_idxs)} samples in {split} set.")

    def __len__(self):
        return len(self.room_idxs)

    def __getitem__(self, idx):
        # 1) Decide which room to load
        room_idx = self.room_idxs[idx]
        room_path = self.room_files[room_idx]

        # 2) Load exactly that one room from disk, then immediately close
        room_data = np.load(room_path)          # [N, 7]
        points = room_data[:, 0:6]              # [N, 6]
        labels = room_data[:, 6].astype(np.int64)  # [N]
        N_points = points.shape[0]

        # 3) Sample a random block until we find > 1024 points
        while True:
            # pick a random point as center
            rand_idx = np.random.choice(N_points)
            center = points[rand_idx, :3]  # xyz of chosen point
            half = self.block_size / 2.0
            block_min = center - np.array([half, half, 0.0])
            block_max = center + np.array([half, half, 0.0])

            # find all points within that xy‐rectangle
            mask = (
                (points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) &
                (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1])
            )
            point_idxs = np.where(mask)[0]
            if point_idxs.size > 1024:
                break

        # 4) If more than num_point points, sample without replacement; else with replacement
        if point_idxs.size >= self.num_point:
            selected = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected = np.random.choice(point_idxs, self.num_point, replace=True)

        # 5) Normalize and build the 9‐dim feature vector
        sel_pts = points[selected, :].copy()  # shape [num_point, 6]
        # current_points: [num_point, 9]
        current_points = np.zeros((self.num_point, 9), dtype=np.float32)

        # a) normalized xyz ratios
        coord_max = self.room_coord_max[room_idx]
        current_points[:, 6] = sel_pts[:, 0] / coord_max[0]
        current_points[:, 7] = sel_pts[:, 1] / coord_max[1]
        current_points[:, 8] = sel_pts[:, 2] / coord_max[2]

        # b) shift coordinates to center
        sel_pts[:, 0] -= center[0]
        sel_pts[:, 1] -= center[1]

        # c) scale RGB to [0,1]
        sel_pts[:, 3:6] /= 255.0

        # d) fill first 6 columns (xyz + RGB)
        current_points[:, 0:6] = sel_pts

        current_labels = labels[selected]  # shape [num_point]

        # 6) If a transform is provided (e.g. ToTensor, jitter), apply it
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)

        # 7) Free that big array before returning (so Python GC can collect it)
        del room_data
        del points
        del labels

        return current_points, current_labels


if __name__ == '__main__':
    data_root = '/mnt/lustre/zhaohengshuang/dataset/s3dis/trainval_fullarea'
    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01

    point_data = S3DIS(
        split='train',
        data_root=data_root,
        num_point=num_point,
        test_area=test_area,
        block_size=block_size,
        sample_rate=sample_rate,
        transform=None
    )
    print('point data size:', len(point_data))
    print('point data 0 shape:', point_data[0][0].shape)
    print('point label 0 shape:', point_data[0][1].shape)

    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)

    train_loader = torch.utils.data.DataLoader(
        point_data,
        batch_size=16,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    for idx in range(4):
        start = time.time()
        for i, (inp, tgt) in enumerate(train_loader):
            print(f'time: {i+1}/{len(train_loader)} -- {time.time() - start:.3f}')
            start = time.time()