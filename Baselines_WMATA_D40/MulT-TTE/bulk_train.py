import sys
import os
import argparse
import copy
import torch
import torch.multiprocessing as mp

from train.training_main import train_main
print(">>> LOADED FROM:", train_main.__code__.co_filename)


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
        metrics = train_main(job_args)

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
    parser.add_argument('-m', '--model', type=str, help='input the model name', default='MulT_TTE', choices=['MulT_TTE'])
    parser.add_argument('-M', '--mode', type=str, default='train', help='input the process mode', choices=['train', 'resume', 'test'])
    parser.add_argument('-d', '--dataset', type=str, default='toronto_927_2025_Jan_Jun_MulT_TTE', help='input the dataset name')
    parser.add_argument('-i', '--identify', type=str, help='input the specific identification information', default='run')
    

    parser.add_argument('-D', '--device', type=str, help='input the chosen device', default="cuda:0")
    
    parser.add_argument('-o', '--optim', type=str, help='input the chosen optimization function', default="Adam", choices=['Adam'])
    parser.add_argument('-c', '--loss', type=str, help='input the chosen loss function', default="smoothL1", choices=['rmse','mse', 'mape', 'mae', 'smoothL1'])
    parser.add_argument('-cl', '--loss_val', type=float, help='intput the specific parameter for smoothL1',  default=300.)
    parser.add_argument('-e', '--epochs', type=int, help='input the max epochs',default=50)
    parser.add_argument('-b', '--beta', type=float, help='intput the learning preference between MSG and TTE (the bigger the value, the more preference for TTE.)',default=0.7)
    parser.add_argument('-l', '--lr', type=float, help='intput the initial learning rate',default=0.001)
    parser.add_argument('-w', '--weight_decay', type=float, help='intput the weight decay of optimization',default=0.00001)
    parser.add_argument('-p', '--patience', type=int, help='intput the max iteration times of early stop',default=10)
    parser.add_argument('-r', '--mask_rate', type=float, help='intput the mask rate of segments in a trajectory', default=0.4)

    base_args = parser.parse_args()
    base_args.absPath = os.path.dirname(os.path.abspath(__file__))

    # 2. Define the different configurations/jobs you want to run in parallel
    # Example: Run 2 jobs across 2 different GPUs with different learning rates
    job_configs = [
        {"device": "cuda:0", 
         "dataset": "toronto_927_2025_Jan_Jun_MulT_TTE",
         "identify": "job_gpu0_lr0005"
         },
        {"device": "cuda:0", 
         "dataset": "toronto_927_2025_Jul_Dec_MulT_TTE",
         "identify": "job_gpu1_lr0005"
         },
        {"device": "cuda:0", 
         "dataset": "toronto_927_2026_Jan_Jun_MulT_TTE",
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
    results = []
    for _ in range(len(job_configs)):
        results.append(result_queue.get())
    
    # 4. Wait for all processes to complete
    for p in processes:
        p.join()

    # 5. Collect and print all results
    print("\n" + "="*40 + " ALL RUNS FINISHED " + "="*40)
    while not result_queue.empty():
        res = result_queue.get()
        metrics = res['metrics']
        print(f"\nResults for Job '{res['identify']}' on {res['device']}:")
        
        if metrics is not None and isinstance(metrics, dict):
            print(f"  MAE:  {metrics.get('TEST_MAE')}")
            print(f"  MAPE: {metrics.get('TEST_MAPE')}")
            print(f"  RMSE: {metrics.get('TEST_RMSE')}")
        else:
            print("  Error: train_main did not return a dictionary!")

    sys.exit(0)