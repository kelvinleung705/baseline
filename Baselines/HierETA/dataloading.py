import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Global Link Definition for your bus line (0-indexed segments)
STATIC_LINKS = [
    [0, 1, 2, 3],  # Link 0: Segments 1, 2, 3, 4
    [4],  # Link 1: Segment 5
    [5, 6, 7, 8]  # Link 2: Segments 6, 7, 8, 9
]


class MySet(Dataset):
    def __init__(self, input_file, FLAGS):
        self.FLAGS = FLAGS
        self.root_dir = FLAGS.data_dir
        file_path = os.path.join(self.root_dir, input_file)

        # Load pre-encoded CSV
        self.data = pd.read_csv(file_path, header=None).values.astype(np.float32)
        self.route_num = len(self.data)
        print(f"Loaded dataset '{input_file}' with {self.route_num} trips.")

    def __getitem__(self, idx):
        row = self.data[idx]

        # 1. Global Pre-Encoded Features (Cols 0 to 6)
        global_features = row[0:7]

        # 2. Start Segment (Col 54: 1 to 9 -> convert to 0-indexed 0 to 8)
        start_seg = int(row[54]) - 1

        # 3. All trips terminate at Segment 9 (Index 8)
        traversed_segments = list(range(start_seg, 9))

        # 4. Extract Segment Target Times (Cols 9 to 17) & Segment Info (Cols 18 to 53)
        # All 9 segment times and 9 segment infos loaded for indexing
        all_seg_times = row[9:18]
        all_seg_info = row[18:54].reshape(9, 4)  # 9 segments, 4 features each

        # Calculate GT total ETA time for traversed segments
        gt_eta_time = np.sum(all_seg_times[start_seg:9])

        # 5. Build Hierarchical Link-Segment Structure for HierETA
        segment_list_hier = []
        seg_info_hier = []
        seg_times_hier = []

        for link_idx, link_segs in enumerate(STATIC_LINKS):
            # Keep only segments that are part of this trip
            active_segs = [s for s in link_segs if s in traversed_segments]

            if len(active_segs) > 0:
                segment_list_hier.append(active_segs)
                seg_info_hier.append(all_seg_info[active_segs])
                seg_times_hier.append(all_seg_times[active_segs])

        return {
            "global_features": global_features,
            "segment_list_hier": segment_list_hier,
            "seg_info_hier": seg_info_hier,
            "seg_times_hier": seg_times_hier,
            "gt_eta_time": gt_eta_time
        }

    def __len__(self):
        return self.route_num


def collate_fn(data, FLAGS):
    batch_size = len(data)
    link_num = 3  # Fixed to 3 links
    segment_num = 4  # Max segments per link

    # Target & Global features
    gt_eta_time = torch.FloatTensor([item["gt_eta_time"] for item in data]).unsqueeze(-1)
    global_features = torch.FloatTensor(np.array([item["global_features"] for item in data]))

    # Hierarchical feature tensors for HierETA model
    # Shape: (Batch, Link_Num, Segment_Num, Feature_Dim)
    seg_info_padded = np.zeros((batch_size, link_num, segment_num, 4), dtype=np.float32)

    # Shape: (Batch, Link_Num, Segment_Num)
    seg_times_padded = np.zeros((batch_size, link_num, segment_num), dtype=np.float32)

    # Masks & Sequence Lengths
    segment_mask = np.zeros((batch_size, link_num, segment_num), dtype=np.float32)
    link_mask = np.zeros((batch_size, link_num), dtype=np.float32)

    link_lens = np.zeros(batch_size, dtype=np.int32)
    link_seg_lens = np.zeros((batch_size, link_num), dtype=np.int32)

    for i, item in enumerate(data):
        hier_segs = item["segment_list_hier"]
        hier_info = item["seg_info_hier"]
        hier_times = item["seg_times_hier"]

        num_links = len(hier_segs)
        link_lens[i] = num_links

        # Link indices mapping back to global 3 links
        # If trip skips Link 0, active links shift accordingly
        link_offset = link_num - num_links

        for l_idx in range(num_links):
            actual_link_idx = l_idx + link_offset
            num_segs = len(hier_segs[l_idx])

            link_seg_lens[i, actual_link_idx] = num_segs
            link_mask[i, actual_link_idx] = 1.0

            # Populate Segment Info & Times
            seg_info_padded[i, actual_link_idx, :num_segs, :] = hier_info[l_idx]
            seg_times_padded[i, actual_link_idx, :num_segs] = hier_times[l_idx]
            segment_mask[i, actual_link_idx, :num_segs] = 1.0

    attrs = {
        "gt_eta_time": gt_eta_time,  # (Batch, 1)
        "global_features": global_features,  # (Batch, 7)
        "seg_info": torch.FloatTensor(seg_info_padded),  # (Batch, 3, 4, 4)
        "seg_times": torch.FloatTensor(seg_times_padded),  # (Batch, 3, 4)
        "link_lens": torch.LongTensor(link_lens),  # (Batch,)
        "link_seg_lens": torch.LongTensor(link_seg_lens),  # (Batch, 3)
        "road_segment_mask": torch.FloatTensor(segment_mask),  # (Batch, 3, 4)
        "road_link_mask": torch.FloatTensor(link_mask)  # (Batch, 3)
    }

    return attrs


def get_loader(input_file, FLAGS):
    dataset = MySet(input_file=input_file, FLAGS=FLAGS)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=FLAGS.batch_size,
        shuffle=FLAGS.is_training,
        collate_fn=lambda x: collate_fn(x, FLAGS),
        num_workers=0,
        pin_memory=True
    )
    return data_loader