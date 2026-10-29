import os
import sys
import importlib.util
import copy
from sklearn import metrics
import torch
import torch.multiprocessing as mp
import pandas as pd

# -------------------------------------------------------------------------
# 1. DYNAMIC IMPORT: preprocess_csv 3.py (handles filename with space)
# -------------------------------------------------------------------------
preprocess_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "algorithm", "gmdnet", "preprocess_csv 3.py")
)

spec = importlib.util.spec_from_file_location("preprocess_csv 3", preprocess_path)
preprocess_module = importlib.util.module_from_spec(spec)
sys.modules["preprocess_csv 3"] = preprocess_module
spec.loader.exec_module(preprocess_module)

# Retrieve the preprocessing function from the imported file
process_three_csvs = preprocess_module.process_three_csvs

# -------------------------------------------------------------------------
# 2. IMPORT DIRECTLY FROM train.py
# -------------------------------------------------------------------------
from algorithm.gmdnet.train import main, get_params, seed_it


# -------------------------------------------------------------------------
# 3. WORKER PIPELINE
# -------------------------------------------------------------------------
def worker_task(config):
    """
    1. Preprocesses the 3 CSVs using process_three_csvs from preprocess_csv 3.py
    2. Overwrites training parameters
    3. Runs train.py's main(params)
    """
    seed_it(1024)
    run_id = config.get("run_id", "Worker")
    
    dataset_name = config["dataset"]
    target_folder = os.path.join("data", dataset_name)
    os.makedirs(target_folder, exist_ok=True)

    print(f"\n[{run_id}] Step 1: Preprocessing data into '{target_folder}'...")
    # Call the exact function from preprocess_csv 3.py
    process_three_csvs(
        train_csv_path=config["train_csv"],
        val_csv_path=config["val_csv"],
        test_csv_path=config["test_csv"],
        json_path=config["json_path"],
        output_dir=target_folder
    )

    print(f"[{run_id}] Step 2: Preparing parameters and starting training...")
    # Get standard defaults from train.py
    params = vars(get_params())

    # Override defaults with custom dictionary configuration
    params.update(config)

    # Point dataset loaders directly to the newly generated .npy files
    params['train_path'] = os.path.join(target_folder, "train.npy")
    params['val_path']   = os.path.join(target_folder, "val.npy")
    params['test_path']  = os.path.join(target_folder, "test.npy")

    if "device" in config:
        params['device'] = torch.device(config["device"])

    # Directly trigger train.py pipeline
    results =main(params)
    
    
    # 2. Specify the exact keys you want to keep:
    metric_keys = ['model', 'val_mae', 'val_mape', 'val_rmse', 'val_log-likelihood', 'val_crps']

    # 3. Extract only those keys into a new dictionary
    metrics_only = {key: results[key] for key in metric_keys if key in results}

    # 4. Convert to DataFrame (wrapping in [ ] creates 1 row where keys are headers)
    df = pd.DataFrame([metrics_only])

    # 5. Save to CSV
    csv_path = os.path.join(target_folder, "results.csv")
    df.to_csv(csv_path, index=False)

    print(f"Metrics saved to {csv_path}")
    
        
    
    print(f"[{run_id}] Finished successfully.")


# -------------------------------------------------------------------------
# 4. MAIN MULTIPROCESSING LAUNCHER
# -------------------------------------------------------------------------
if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)

    JSON_PATH = "segments_toronto.json"

    # Define your 3 experiments/workers
    tasks = [
            {
                "run_id": "Phase_1_2025_1-6",
                "train_csv": "data/phase_1/TTC_927_Period_1_train.csv",
                "val_csv":   "data/phase_1/TTC_927_Period_1_validation.csv",
                "test_csv":  "data/phase_1/TTC_927_Period_1_test.csv",
                "json_path": JSON_PATH,
                "dataset": "phase_1",       # Preprocessing destination
                "batch_size": 32,
                "num_epoch": 100,
                "lr": 0.001,
                "device": "cuda:0",
            },
            {
                "run_id": "Phase_2_2025_7-12",
                "train_csv": "data/phase_2/TTC_927_Period_2_train.csv",
                "val_csv":   "data/phase_2/TTC_927_Period_2_validation.csv",
                "test_csv":  "data/phase_2/TTC_927_Period_2_test.csv",
                "json_path": JSON_PATH,
                "dataset": "phase_2",       # Preprocessing destination
                "batch_size": 32,
                "num_epoch": 100,
                "lr": 0.001,
                "device": "cuda:0",                # Change to cuda:1 if multiple GPUs available
            },
            {
                "run_id": "Phase_3_2026_1-6",
                "train_csv": "data/phase_3/TTC_927_Period_3_train.csv",
                "val_csv":   "data/phase_3/TTC_927_Period_3_validation.csv",
                "test_csv":  "data/phase_3/TTC_927_Period_3_test.csv",
                "json_path": JSON_PATH,
                "dataset": "phase_3",       # Preprocessing destination
                "batch_size": 32,
                "num_epoch": 100,
                "lr": 0.001,
                "device": "cuda:0",
            },
        ]

    max_workers = min(len(tasks), 3)
    print(f"Spawning {len(tasks)} parallel runs across {max_workers} processes...")

    with mp.Pool(processes=max_workers) as pool:
        pool.map(worker_task, tasks)

    print("All tasks have finished.")
    print("All preparation and training processes have completed.")