import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class BusLineDataset(Dataset):
    def __init__(self, data_dir, input_file, is_training=True):
        """
        Loads the pre-encoded CSV dataset.
        """
        self.is_training = is_training
        file_path = os.path.join(data_dir, input_file)

        # Load CSV using pandas for fast parsing (Assuming no header, if you have a header add header=0)
        self.data = pd.read_csv(file_path, header=None).values.astype(np.float32)
        self.route_num = len(self.data)
        print(f"Loaded {input_file}: {self.route_num} trips.")

    def __getitem__(self, idx):
        row = self.data[idx]

        # --- 1. Global Context Features (Indices 0 to 6) ---
        time_enc = row[0:2]
        weekday = row[2:3]
        weekend = row[3:4]
        month = row[4:6]
        holiday = row[6:7]
        global_features = np.concatenate([time_enc, weekday, weekend, month, holiday])  # shape (7,)

        # --- 2. Trip Routing Info ---
        # Assuming CSV says "1" for Segment 1. We subtract 1 for Python 0-indexing.
        start_seg_idx = int(row[54]) - 1
        trip_len = int(row[55])

        # List of actual segment indices traversed in this trip (e.g., [2, 3, 4] if starting at seg 3 for 3 segs)
        traversed_segments = list(range(start_seg_idx, start_seg_idx + trip_len))

        LINK_MAP = [[0, 1, 2, 3], [4], [5, 6, 7, 8]]
        segment_list_hier = []
        for link_segs in LINK_MAP:
            # Keep only segments that are actually in this trip
            valid_segs_in_link = [s for s in link_segs if s in traversed_segments]
            if valid_segs_in_link:
                segment_list_hier.append(valid_segs_in_link)


        # --- 3. Sequence Target (Travel Time per Segment) ---
        # Slicing the exact segments traversed for this specific trip
        time_slice_start = 9 + start_seg_idx
        time_slice_end = time_slice_start + trip_len

        seg_times = row[time_slice_start: time_slice_end]
        gt_eta_time = np.sum(seg_times)  # Total trip time

        # --- 4. Sequence Features (Condition/Speed) ---
        # Slicing the exact segment info features traversed for this specific trip
        # Each segment has 4 features (N+0 to N+3)
        info_slice_start = 18 + (start_seg_idx * 4)
        info_slice_end = info_slice_start + (trip_len * 4)

        # Shape: (trip_len, 4)
        seg_info = row[info_slice_start: info_slice_end].reshape(trip_len, 4)

        return {
            "global_features": global_features,
            "seg_info": seg_info,
            "seg_times": seg_times,
            "gt_eta_time": gt_eta_time,
            "trip_len": trip_len,
            "segment_list_hier": segment_list_hier
        }

    def __len__(self):
        return self.route_num


def collate_fn(batch):
    """
    Pads the sequence data (seg_info) to the maximum length in the current batch.
    Creates masks for the model to ignore padded steps.
    """
    batch_size = len(batch)

    # 1. Extract non-sequence data
    global_feats = [item["global_features"] for item in batch]
    gt_eta = [item["gt_eta_time"] for item in batch]
    trip_lens = [item["trip_len"] for item in batch]

    # Max length in THIS batch (up to 9)
    max_len = max(trip_lens)

    # 2. Initialize padded tensors
    padded_seg_info = np.zeros((batch_size, max_len, 4), dtype=np.float32)
    padded_seg_times = np.zeros((batch_size, max_len), dtype=np.float32)
    mask = np.zeros((batch_size, max_len), dtype=np.float32)

    # 3. Fill padded tensors
    for i, item in enumerate(batch):
        t_len = item["trip_len"]
        padded_seg_info[i, :t_len, :] = item["seg_info"]
        padded_seg_times[i, :t_len] = item["seg_times"]
        mask[i, :t_len] = 1.0  # 1 means valid data, 0 means padded padding

    return {
        "global_features": torch.FloatTensor(np.array(global_feats)),  # Shape: (Batch, 7)
        "seg_info": torch.FloatTensor(padded_seg_info),  # Shape: (Batch, max_len, 4)
        "seg_times": torch.FloatTensor(padded_seg_times),  # Shape: (Batch, max_len)
        "gt_eta_time": torch.FloatTensor(np.array(gt_eta)).unsqueeze(-1),  # Shape: (Batch, 1)
        "seq_lens": torch.LongTensor(np.array(trip_lens)),  # Shape: (Batch,)
        "mask": torch.FloatTensor(mask)  # Shape: (Batch, max_len)
    }


def get_loader(data_dir, input_file, batch_size, is_training=True):
    dataset = BusLineDataset(data_dir=data_dir, input_file=input_file, is_training=is_training)

    # Shuffling natively supported by DataLoader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_training,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    return data_loader