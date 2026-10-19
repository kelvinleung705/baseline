import os
import json
import numpy as np
import pandas as pd


def generate_data_info(data_dir="./samples/",
                       train_file="train_trips.csv",
                       test_file="test_trips.csv",
                       network_file="data-info/WMATA_D40_road_network.json",
                       output_file="data-info/data_info.json"):
    # -------------------------------------------------------------
    # 1. Read static features from data-info/road_network.json
    # -------------------------------------------------------------
    if not os.path.exists(network_file):
        raise FileNotFoundError(f"Cannot find '{network_file}'. Please create it first!")

    with open(network_file, 'r') as f:
        network_data = json.load(f)

    segments = network_data["segments"]
    lens = [s["length_m"] for s in segments]
    speed_lims = [s["max_speed_kmh"] for s in segments]
    wids = [s["width_lanes"] for s in segments]

    # Calculate static map statistics
    len_mean, len_std = float(np.mean(lens)), float(np.std(lens))
    speed_lim_mean, speed_lim_std = float(np.mean(speed_lims)), float(np.std(speed_lims))
    wid_mean, wid_std = float(np.mean(wids)), float(np.std(wids))

    # -------------------------------------------------------------
    # 2. Function to compute gt_eta_time stats from CSV using your loop
    # -------------------------------------------------------------
    def get_gt_eta_stats(csv_path):
        data = pd.read_csv(csv_path).to_numpy().astype(np.float32)

        start_segs = (data[:, 75] - 1).astype(int)
        print(start_segs)
        gt_eta_times = []
        for i in range(len(data)):
            s = start_segs[i]
            gt_eta_times.append(np.sum(data[i, 9:22][s:13]))

        return float(np.mean(gt_eta_times)), float(np.std(gt_eta_times))

    train_gt_mean, train_gt_std = get_gt_eta_stats(os.path.join(data_dir, train_file))
    test_gt_mean, test_gt_std = get_gt_eta_stats(os.path.join(data_dir, test_file))

    # -------------------------------------------------------------
    # 3. Assemble output JSON
    # -------------------------------------------------------------
    data_info = {
        "train_len_mean": round(len_mean, 4),
        "train_len_std": round(len_std, 4),
        "train_speed_lim_mean": round(speed_lim_mean, 4),
        "train_speed_lim_std": round(speed_lim_std, 4),
        "train_wid_mean": round(wid_mean, 4),
        "train_wid_std": round(wid_std, 4),
        "train_gt_eta_time_mean": round(train_gt_mean, 4),
        "train_gt_eta_time_std": round(train_gt_std, 4),

        "test_len_mean": round(len_mean, 4),
        "test_len_std": round(len_std, 4),
        "test_speed_lim_mean": round(speed_lim_mean, 4),
        "test_speed_lim_std": round(speed_lim_std, 4),
        "test_wid_mean": round(wid_mean, 4),
        "test_wid_std": round(wid_std, 4),
        "test_gt_eta_time_mean": round(test_gt_mean, 4),
        "test_gt_eta_time_std": round(test_gt_std, 4)
    }

    # Save to data-info/data_info.json
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(data_info, f, indent=2)

    print(f"Successfully generated '{output_file}'!")
    print(json.dumps(data_info, indent=2))


if __name__ == '__main__':
    generate_data_info()