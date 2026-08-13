import os
import json
import numpy as np
import pandas as pd


def parse_csv(csv_path, segment_map, num_nodes):
    """
    Reads a CSV and converts it into GMDNet numpy arrays.
    """
    has_header = pd.read_csv(csv_path, nrows=1).columns[0].isalpha()
    df = pd.read_csv(csv_path, header=0 if has_header else None)
    data_mat = df.to_numpy(dtype=np.float32)

    num_samples = len(data_mat)
    max_seq_len = 9  # Maximum trip length in segments

    # 1. Extract columns
    temporal_features = data_mat[:, 0:7]

    sin_t, cos_t = data_mat[:, 0], data_mat[:, 1]
    angles = np.arctan2(sin_t, cos_t)
    hours = np.mod(np.round((angles % (2 * np.pi)) / (2 * np.pi) * 24.0), 24).astype(np.int64)

    seg_travel_times = data_mat[:, 9:18]
    seg_conditions = data_mat[:, 19:55].reshape(num_samples, 9, 4)

    start_segs = data_mat[:, 55].astype(int)
    trip_lens = data_mat[:, 56].astype(int)

    # 2. Initialize arrays
    routes = np.zeros((num_samples, max_seq_len, 2), dtype=np.int64)
    masks = np.zeros((num_samples, max_seq_len, max_seq_len), dtype=np.int64)
    labels = np.zeros((num_samples, 1), dtype=np.float32)
    f_features = np.zeros((num_samples, 1 + 1 + 7), dtype=np.float32)

    edges = np.zeros((num_samples, num_nodes, num_nodes, 6), dtype=np.float32)
    nodes = np.zeros((num_samples, num_nodes, 4), dtype=np.float32)
    adjacency = np.zeros((num_samples, num_nodes, num_nodes), dtype=np.int64)

    # 3. Populate arrays
    for i in range(num_samples):
        start_seg_id = start_segs[i]
        t_len = min(trip_lens[i], 9 - start_seg_id + 1)
        s_idx = start_seg_id - 1

        labels[i, 0] = np.sum(seg_travel_times[i, s_idx: s_idx + t_len])

        for step in range(t_len):
            curr_seg_id = start_seg_id + step
            u, v = segment_map[curr_seg_id]
            routes[i, step, 0] = u
            routes[i, step, 1] = v

        masks[i, :t_len, :t_len] = 1

        f_features[i, 1] = hours[i]
        f_features[i, 2:] = temporal_features[i]

        for u in range(num_nodes):
            nodes[i, u, 0] = u
            nodes[i, u, 1:] = 1.0

        for seg_id, (u, v) in segment_map.items():
            arr_idx = seg_id - 1
            adjacency[i, u, v] = 1
            edges[i, u, v, 0] = u
            edges[i, u, v, 1] = v
            edges[i, u, v, 2:] = seg_conditions[i, arr_idx, :]

    return {
        'route': routes, 'mask': masks, 'f': f_features,
        'label': labels, 'edge': edges, 'node': nodes, 'A': adjacency
    }


def process_two_csvs(train_csv_path, test_csv_path, json_path, output_dir="./data", train_ratio=0.85):
    """
    Splits train_csv into Train/Val, and saves test_csv entirely as Test.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load JSON Topology
    with open(json_path, 'r') as f:
        segment_map_raw = json.load(f)
    segment_map = {int(k): v for k, v in segment_map_raw.items()}

    num_nodes = max([max(u, v) for u, v in segment_map.values()]) + 1

    # Parse both CSVs
    print("Parsing Training/Validation CSV...")
    train_val_data = parse_csv(train_csv_path, segment_map, num_nodes)

    print("Parsing Test CSV...")
    test_data = parse_csv(test_csv_path, segment_map, num_nodes)

    # Split Train/Val Data
    num_train_val = len(train_val_data['label'])
    indices = np.arange(num_train_val)
    np.random.seed(1024)
    np.random.shuffle(indices)

    train_end = int(num_train_val * train_ratio)

    splits = {
        'train': indices[:train_end],
        'val': indices[train_end:],
    }

    # Save Train and Val
    for mode, split_idx in splits.items():
        split_size = len(split_idx)

        data_dict = {}
        for key in train_val_data.keys():
            data_dict[key] = train_val_data[key][split_idx].copy()

        # VERY IMPORTANT: Reset f[:, 0] to local batch index (0 to N-1) for GMDNet lookups
        data_dict['f'][:, 0] = np.arange(split_size)

        save_path = os.path.join(output_dir, f"{mode}.npy")
        np.save(save_path, data_dict)
        print(f"Saved {mode} set -> {save_path} | Samples: {split_size}")

    # Save Test directly from test_data
    test_size = len(test_data['label'])
    test_data['f'][:, 0] = np.arange(test_size)  # Reset index for test data too

    test_save_path = os.path.join(output_dir, "test.npy")
    np.save(test_save_path, test_data)
    print(f"Saved test set -> {test_save_path} | Samples: {test_size}")


if __name__ == "__main__":
    TRAIN_CSV = "train_dataset.csv"  # File to be split into train.npy and val.npy
    TEST_CSV = "test_dataset.csv"  # File to become test.npy directly
    JSON_PATH = "segments.json"

    process_two_csvs(TRAIN_CSV, TEST_CSV, JSON_PATH)