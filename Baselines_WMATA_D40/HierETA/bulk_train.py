import sys
import os
import argparse
import copy
import torch
import torch.multiprocessing as mp

from main_bulk import run_experiment
print(">>> LOADED FROM:", run_experiment.__code__.co_filename)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def worker(job_args, result_queue):
    """
    Worker function executed by each spawned process.
    """
    try:
        # Explicitly set the active CUDA device for this process
        if "cuda" in job_args.device:
            torch.cuda.set_device(job_args.device)

        print(f"--> [PID {os.getpid()}] Starting run on {job_args.device} (ID: {job_args.identify})")

        # Run the training logic
        metrics = run_experiment(job_args)

        # Send results back
        result_queue.put({
            'identify': job_args.identify,
            'device': job_args.device,
            'metrics': metrics,
            'error': None
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        result_queue.put({
            'identify': job_args.identify,
            'device': job_args.device,
            'metrics': None,
            'error': str(e)
        })


if __name__ == '__main__':
    # 1. CRITICAL: CUDA requires the 'spawn' start method in PyTorch multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--is_training", type=str2bool, default=True, help="training mode or not",)
    parser.add_argument("--segment_num", type=int, default=4, help="segment number per link")
    parser.add_argument("--link_num", type=int, default=3, help="link number per route")
    parser.add_argument("--win_size", type=int, default=3, help="window scale of neighboring segments",)
    parser.add_argument("--Lambda", type=float, default=0.4, help="weighting parameter in decoder",)
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")

    # Dataset directory and CSV file names
    parser.add_argument("--data_dir", type=str, default="./data-info_phrase_1/", help="directory for route data storage")
    parser.add_argument("--train_file", type=str, default="train_trips.csv", help="training csv filename")
    parser.add_argument("--eval_file", type=str, default="validation_trips.csv", help="evaluation csv filename")
    parser.add_argument("--test_file", type=str, default="test_trips.csv", help="test csv filename")

    parser.add_argument("--log_dir", type=str, default="data-info_phrase_1/logs", help="directory for logging")
    parser.add_argument("--step_per_eval", type=int, default=100, help="training steps per evaluation")
    parser.add_argument("--use_tb", type=str2bool, default=False, help="Use tensorboard to log training info",)
    parser.add_argument("--code_backup", type=str2bool, default=True, help="code backup or not")
    parser.add_argument("--description", type=str, default="HierETA", help="description of current running experiments.")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to checkpoint folder or .pth file for testing",)

    base_args = parser.parse_args()
    base_args.absPath = os.path.dirname(os.path.abspath(__file__))

    # 2. Define the different configurations/jobs you want to run in parallel
    # Example: Run 2 jobs across 2 different GPUs with different learning rates
    job_configs = [
        {"device": "cuda:0", 
         "data_dir": "./data-info_phrase_1/",
         "log_dir": "./data-info_phrase_1/logs",
         "identify": "job_gpu0_lr0005"
         },
        {"device": "cuda:0", 
         "data_dir": "./data-info_phrase_2/",
         "log_dir": "./data-info_phrase_2/logs",
         "identify": "job_gpu1_lr0005"
         },
        {"device": "cuda:0", 
         "data_dir": "./data-info_phrase_3/",
         "log_dir": "./data-info_phrase_3/logs",
         "identify": "job_gpu2_lr0005"
        }
    ]

    # Queue to collect metrics returned from each process
    result_queue = mp.Queue()
    processes = []

    # 3. Spawn and start each process
    for config in job_configs:
        # Create a deep copy so arguments don't overwrite each other
        job_args = copy.deepcopy(base_args)
        
        # Override specific arguments for this job
        for key, val in config.items():
            setattr(job_args, key, val)

        p = mp.Process(target=worker, args=(job_args, result_queue))
        p.start()
        processes.append(p)

    # 4. Drain the queue BEFORE or DURING join to prevent multiprocessing deadlocks
    collected_results = []
    for _ in range(len(job_configs)):
        collected_results.append(result_queue.get())
    
    # 4. Wait for all processes to complete
    for p in processes:
        p.join()

    # 5. Collect and print all results
    print("\n" + "="*40 + " ALL RUNS FINISHED " + "="*40)
    for res in collected_results:
        metrics = res['metrics']
        print(f"\nResults for Job '{res['identify']}' on {res['device']}:")
        
        if res['error']:
            print(f"  Failed with Error: {res['error']}")
        elif metrics is not None and isinstance(metrics, dict):
            # 【關鍵修復】：使用小寫鍵值匹配 main_bulk.py 的回傳格式
            print(f"  Test MAE:  {metrics.get('test_mae', metrics.get('mae'))}")
            print(f"  Test MAPE: {metrics.get('test_mape', metrics.get('mape'))}")
            print(f"  Test RMSE: {metrics.get('test_rmse', metrics.get('rmse'))}")
            print(f"  Checkpoint: {metrics.get('best_model_path')}")
        else:
            print("  Error: run_experiment did not return valid metrics!")

    sys.exit(0)