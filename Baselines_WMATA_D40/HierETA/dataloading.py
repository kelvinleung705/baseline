import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import utils


def load_network_data(network_file_path):
    if not os.path.exists(network_file_path):
        raise FileNotFoundError(f"Cannot find network file at: '{network_file_path}'")

    with open(network_file_path, 'r') as f:
        network_data = json.load(f)

    static_links = [link["segments"] for link in network_data["links"]]
    segments_sorted = sorted(network_data["segments"], key=lambda s: s["segment_id"])

    # Extract static metadata per segment
    static_dict = {
        "len": np.array([s["length_m"] for s in segments_sorted], dtype=np.float32),
        "speedLimit": np.array([s["max_speed_kmh"] for s in segments_sorted], dtype=np.float32),
        "wid": np.array([s["width_lanes"] for s in segments_sorted], dtype=np.float32),
        "segment_functional_level": np.array([s.get("functional_level", 1) for s in segments_sorted], dtype=np.int64),
        "roadLevel": np.array([s.get("road_level", 1) for s in segments_sorted], dtype=np.int64),
        "laneNum": np.array([s["width_lanes"] for s in segments_sorted], dtype=np.int64)
    }

    return static_links, static_dict


class MySet(Dataset):
    def __init__(self, input_file, FLAGS):
        self.FLAGS = FLAGS
        self.root_dir = FLAGS.data_dir
        self.is_training = FLAGS.is_training

        base_dir = os.path.dirname(os.path.abspath(__file__))
        network_file_path = getattr(
            FLAGS,
            'network_file',
            os.path.join(base_dir, "data-info", "toronto_927_900_road_network.json")
        )

        self.static_links, self.static_meta = load_network_data(network_file_path)

        # Normalize static continuous features
        self.static_meta["len_norm"] = np.array(
            [utils.normalize(x, "len", self.is_training) for x in self.static_meta["len"]], dtype=np.float32)
        self.static_meta["speedLimit_norm"] = np.array(
            [utils.normalize(x, "speed_lim", self.is_training) for x in self.static_meta["speedLimit"]],
            dtype=np.float32)
        self.static_meta["wid_norm"] = np.array(
            [utils.normalize(x, "wid", self.is_training) for x in self.static_meta["wid"]], dtype=np.float32)

        file_path = os.path.join(self.root_dir, input_file)
        self.data = pd.read_csv(file_path).to_numpy().astype(np.float32)
        self.route_num = len(self.data)
        print(f"Loaded dataset '{input_file}' with {self.route_num} trips.")

    def __getitem__(self, idx):
        row = self.data[idx]

        global_features = row[0:7]

        start_seg = int(row[54]) - 1
        traversed_segments = list(range(start_seg, 9))

        all_seg_times = row[9:18]
        live_seg_info = row[18:54].reshape(9, 4)

        gt_eta_time = np.sum(all_seg_times[start_seg:9])

        segment_list_hier = []
        seg_times_hier = []
        seg_road_state_hier = []

        for link_idx, link_segs in enumerate(self.static_links):
            active_segs = [s for s in link_segs if s in traversed_segments]
            if len(active_segs) > 0:
                segment_list_hier.append(active_segs)
                seg_times_hier.append(all_seg_times[active_segs])
                seg_road_state_hier.append(live_seg_info[active_segs, 0].astype(np.int64))

        return {
            "global_features": global_features,
            "segment_list_hier": segment_list_hier,
            "seg_times_hier": seg_times_hier,
            "seg_road_state_hier": seg_road_state_hier,
            "gt_eta_time": gt_eta_time
        }

    def __len__(self):
        return self.route_num


def collate_fn(data, FLAGS, static_links, static_meta):
    batch_size = len(data)
    link_num = len(static_links)
    segment_num = max(len(link) for link in static_links)
    total_segs = link_num * segment_num  # 3 * 4 = 12 total flattened slots

    gt_eta_time = torch.FloatTensor([item["gt_eta_time"] for item in data]).unsqueeze(-1)
    global_features = torch.FloatTensor(np.array([item["global_features"] for item in data]))

    # Flattened 2D matrices (Batch, 12) expected by original GitHub Attr class
    seg_id_padded = np.zeros((batch_size, total_segs), dtype=np.int64)
    seg_func_padded = np.zeros((batch_size, total_segs), dtype=np.int64)
    road_state_padded = np.zeros((batch_size, total_segs), dtype=np.float32)
    lane_num_padded = np.zeros((batch_size, total_segs), dtype=np.int64)
    road_level_padded = np.zeros((batch_size, total_segs), dtype=np.int64)

    wid_padded = np.zeros((batch_size, total_segs), dtype=np.float32)
    speed_lim_padded = np.zeros((batch_size, total_segs), dtype=np.float32)
    time_padded = np.zeros((batch_size, total_segs), dtype=np.float32)
    len_padded = np.zeros((batch_size, total_segs), dtype=np.float32)

    # Hierarchical 3D masks for HierETA model decoder
    seg_times_hier_padded = np.zeros((batch_size, link_num, segment_num), dtype=np.float32)
    segment_mask = np.zeros((batch_size, link_num, segment_num), dtype=np.float32)
    link_mask = np.zeros((batch_size, link_num), dtype=np.float32)

    link_lens = np.zeros(batch_size, dtype=np.int32)
    link_seg_lens = np.zeros((batch_size, link_num), dtype=np.int32)

    for i, item in enumerate(data):
        hier_segs = item["segment_list_hier"]
        hier_times = item["seg_times_hier"]
        hier_road_state = item["seg_road_state_hier"]

        num_links = len(hier_segs)
        link_lens[i] = num_links
        link_offset = link_num - num_links

        for l_idx in range(num_links):
            actual_link_idx = l_idx + link_offset
            segs = hier_segs[l_idx]
            num_segs = len(segs)

            link_seg_lens[i, actual_link_idx] = num_segs
            link_mask[i, actual_link_idx] = 1.0

            # Calculate 1D index offset inside the 12-slot flat sequence
            flat_start_idx = actual_link_idx * segment_num

            # Populate 2D Flat matrices (Batch, 12) for Attr
            seg_id_padded[i, flat_start_idx: flat_start_idx + num_segs] = [s + 1 for s in segs]
            seg_func_padded[i, flat_start_idx: flat_start_idx + num_segs] = static_meta["segment_functional_level"][
                segs]
            lane_num_padded[i, flat_start_idx: flat_start_idx + num_segs] = static_meta["laneNum"][segs]
            road_level_padded[i, flat_start_idx: flat_start_idx + num_segs] = static_meta["roadLevel"][segs]
            road_state_padded[i, flat_start_idx: flat_start_idx + num_segs] = (
                hier_road_state[l_idx]
            )

            wid_padded[i, flat_start_idx: flat_start_idx + num_segs] = static_meta["wid_norm"][segs]
            speed_lim_padded[i, flat_start_idx: flat_start_idx + num_segs] = static_meta["speedLimit_norm"][segs]
            len_padded[i, flat_start_idx: flat_start_idx + num_segs] = static_meta["len_norm"][segs]
            time_padded[i, flat_start_idx: flat_start_idx + num_segs] = hier_times[l_idx]

            # Populate 3D matrices (Batch, 3, 4) for HierETA decoder
            seg_times_hier_padded[i, actual_link_idx, :num_segs] = hier_times[l_idx]
            segment_mask[i, actual_link_idx, :num_segs] = 1.0

    week_id = global_features[:, 0].long()
    time_id = global_features[:, 1].long()
    driver_id = torch.zeros(batch_size, dtype=torch.long)  # Dummy driverID

    cross_id = torch.zeros((batch_size, link_num), dtype=torch.long)
    delay_time = torch.zeros((batch_size, link_num), dtype=torch.float32)

    attrs = {
        "ext": global_features,
        "global_features": global_features,
        # seg_cates (Integers)
        "segID": torch.LongTensor(seg_id_padded),
        "segment_functional_level": torch.LongTensor(seg_func_padded),
        "laneNum": torch.LongTensor(lane_num_padded),
        "roadLevel": torch.LongTensor(road_level_padded),
        # seg_conts (Floats)
        "wid": torch.FloatTensor(wid_padded),
        "speedLimit": torch.FloatTensor(speed_lim_padded),
        "time": torch.FloatTensor(time_padded),
        "len": torch.FloatTensor(len_padded),
        "roadState": torch.FloatTensor(road_state_padded),  # <--- Now FloatTensor!
        # link_cates & link_conts
        "crossID": cross_id,
        "delayTime": delay_time,
        # HierETA Masks & Targets
        "gt_eta_time": gt_eta_time,
        "seg_times": torch.FloatTensor(seg_times_hier_padded),
        "link_lens": torch.LongTensor(link_lens),
        "link_seg_lens": torch.LongTensor(link_seg_lens),  # <--- FIXES KeyError: 'link_seg_lens'
        "road_segment_mask": torch.FloatTensor(segment_mask),
        "road_link_mask": torch.FloatTensor(link_mask),
    }
    return attrs


def get_loader(input_file, FLAGS):
    dataset = MySet(input_file=input_file, FLAGS=FLAGS)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=FLAGS.batch_size,
        shuffle=FLAGS.is_training,
        collate_fn=lambda x: collate_fn(x, FLAGS, dataset.static_links, dataset.static_meta),
        num_workers=0,
        pin_memory=True,
        drop_last=FLAGS.is_training
    )
    return data_loader