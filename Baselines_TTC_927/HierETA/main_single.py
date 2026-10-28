import argparse
import copy
import json
import math
import os
import sys

# Set GPU environment variable BEFORE importing PyTorch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.optim as optim

torch.backends.cudnn.enabled = False

import dataloading
from log import logger_tb, message_logger
from models import HierETA
import utils


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
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
    parser.add_argument("--data_dir", type=str, default="./samples/", help="directory for route data storage")
    parser.add_argument("--train_file", type=str, default="train_trips.csv", help="training csv filename")
    parser.add_argument("--eval_file", type=str, default="validation_trips.csv", help="evaluation csv filename")
    parser.add_argument("--test_file", type=str, default="test_trips.csv", help="test csv filename")

    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--step_per_eval", type=int, default=100, help="training steps per evaluation")
    parser.add_argument("--use_tb", type=str2bool, default=False, help="Use tensorboard to log training info",)
    parser.add_argument("--code_backup", type=str2bool, default=True, help="code backup or not")
    parser.add_argument("--description", type=str, default="HierETA", help="description of current running experiments.")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to checkpoint folder or .pth file for testing",)
    return parser


def evaluate(model, files, flags, device):
    model.eval()
    MAE_loss = []
    MAPE_loss = []
    MSE_loss = []

    for file_idx, input_file in enumerate(files):
        MAE_loss_single = []
        MAPE_loss_single = []
        MSE_loss_single = []

        data_iter = dataloading.get_loader(input_file, flags)
        for idx, attr in enumerate(data_iter):
            attr = utils.to_var(attr, device)
            pred, label = model(attr)

            mae = utils.MAE(pred, label)
            mape = utils.MAPE(pred, label)
            rmse = utils.RMSE(pred, label)

            MAE_loss_single.append(mae.item())
            MAPE_loss_single.append(mape.item())
            MSE_loss_single.append(rmse.item() ** 2)

            if idx > 10 and idx % 100 == 0:
                cur_mae = sum(MAE_loss_single) / len(MAE_loss_single)
                cur_mape = sum(MAPE_loss_single) / len(MAPE_loss_single)
                cur_rmse = math.sqrt(
                    sum(MSE_loss_single) / len(MSE_loss_single)
                )
                print(
                    "Evaluate Progress: {:.2f}% | Step: {} | MAE: {:.5f} | MAPE: {:.5f} | RMSE: {:.5f}".format(
                        (idx + 1) * 100 / len(data_iter),
                        idx,
                        cur_mae,
                        cur_mape,
                        cur_rmse,
                    )
                )

        single_file_mae = (
            sum(MAE_loss_single) / len(MAE_loss_single)
            if MAE_loss_single
            else 0.0
        )
        single_file_mape = (
            sum(MAPE_loss_single) / len(MAPE_loss_single)
            if MAPE_loss_single
            else 0.0
        )
        single_file_rmse = (
            math.sqrt(sum(MSE_loss_single) / len(MSE_loss_single))
            if MSE_loss_single
            else 0.0
        )
        """
        print("***********************")
        print(f"File: {input_file}")
        print(f"MAE_loss : {single_file_mae:.5f}")
        print(f"MAPE_loss: {single_file_mape:.5f}")
        print(f"RMSE_loss: {single_file_rmse:.5f}")
        print("***********************\n")
        """

        MAE_loss.extend(MAE_loss_single)
        MAPE_loss.extend(MAPE_loss_single)
        MSE_loss.extend(MSE_loss_single)

    total_mae = sum(MAE_loss) / len(MAE_loss) if MAE_loss else float("inf")
    total_mape = sum(MAPE_loss) / len(MAPE_loss) if MAPE_loss else 0.0
    total_rmse = math.sqrt(sum(MSE_loss) / len(MSE_loss)) if MSE_loss else 0.0

    """
    print("\n-------- Final Evaluation --------")
    print(
        f"MAE : {total_mae:.5f}\nMAPE: {total_mape:.5f}\nRMSE: {total_rmse:.5f}\n"
    )
    """
    return total_mae, total_mape, total_rmse





def train(model, optimizer, flags, device, logger):
    train_set = [flags.train_file]
    eval_set = [flags.eval_file]
    test_set = [flags.test_file]

    print("train file nums: ", len(train_set))
    model.train()
    model.to(device)

    step = 0
    best_mae = float("inf")
    best_mape = float("inf")
    best_rmse = float("inf")
    best_epoch = -1
    best_model_path = os.path.join(logger.log_dir, "best_model.pth")

    for epoch in range(flags.epochs):
        print(f"train files {train_set}")
        print(f"eval files {eval_set}")
        print(f"--- Training epoch {epoch} ---")

        for input_file in train_set:
            model.train()
            print(f"--- Train on file {input_file} ---")

            data_iter = dataloading.get_loader(input_file, flags)
            data_iter_len = len(data_iter)

            for idx, attr in enumerate(data_iter):
                attr = utils.to_var(attr, device)

                optimizer.zero_grad()
                pred, label = model(attr)
                loss = utils.MAE(pred, label)
                loss.backward()
                optimizer.step()
                step += 1

                mape = utils.MAPE(pred, label)
                rmse = utils.RMSE(pred, label)

                if idx and (idx + 1) % 2 == 0:
                    print(
                        "--Progress: {:.4f} step:{} MAE_loss {:.4f} MAPE_loss {:.4f} RMSE_loss {:.4f}".format(
                            (idx + 1) * 100 / data_iter_len,
                            step,
                            loss.item(),
                            mape.item(),
                            rmse.item(),
                        )
                    )

                # Validation
                if step and step % flags.step_per_eval == 0:
                    with torch.no_grad():
                        val_mae, val_mape, val_rmse = evaluate(
                            model, eval_set, flags, device
                        )
                        if val_mae < best_mae:
                            best_mae = val_mae
                            best_mape = val_mape
                            best_rmse = val_rmse
                            best_epoch = epoch
                            check_point = {
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "epoch": epoch,
                                "step": step,
                                "best_mae": best_mae,
                            }
                            print("FINAL TEST RESULTS-New best found:")
                            print(f"Validation MAE : {best_mae:.5f}")
                            print(f"Validation MAPE: {best_mape:.5f}")
                            print(f"Validation RMSE: {best_rmse:.5f}")
                            torch.save(check_point, best_model_path)
                            print(
                                ">>> Best model saved to {} with Val MAE: {:.5f} <<<\n".format(
                                    best_model_path, best_mae
                                )
                            )

                    model.train()

    # Final evaluation on the test set
    print(
        "\n" + "=" * 60 + "\n TRAINING FINISHED. RUNNING EVALUATION ON TEST SET\n"
    )
    if os.path.exists(best_model_path):
        print(f"Loading best checkpoint from: {best_model_path}")
        check_point = torch.load(best_model_path, map_location=device)
        model.load_state_dict(check_point["model"])
    else:
        print("Warning: No checkpoint saved. Testing with current weights.")

    with torch.no_grad():
        test_mae, test_mape, test_rmse = evaluate(
            model, test_set, flags, device
        )
    """
    print("=" * 60)
    print("FINAL TEST RESULTS:")
    print(f"Test MAE : {test_mae:.5f}")
    print(f"Test MAPE: {test_mape:.5f}")
    print(f"Test RMSE: {test_rmse:.5f}")
    print("=" * 60 + "\n")

    return {
        "mae": test_mae,
        "mape": test_mape,
        "rmse": test_rmse,
        "best_model_path": best_model_path,
    }
    """
    print("=" * 60)
    print(f"Selected Epoch: {best_epoch}")
    print("=" * 60)
    print("FINAL Validation RESULTS:")
    print(f"Validation MAE : {best_mae:.5f}")
    print(f"Validation MAPE: {best_mape:.5f}")
    print(f"Validation RMSE: {best_rmse:.5f}")
    print("=" * 60 + "\n")
    
    print("=" * 60)
    print("FINAL Test RESULTS:")
    print(f"Test MAE : {test_mae:.5f}")
    print(f"Test MAPE: {test_mape:.5f}")
    print(f"Test RMSE: {test_rmse:.5f}")
    print("=" * 60 + "\n")
    return {
        "val_mae": best_mae,
        "val_mape": best_mape,
        "val_rmse": best_rmse,
        "test_mae": test_mae,
        "test_mape": test_mape,
        "test_rmse": test_rmse,
        "best_epoch": best_epoch,
        "best_model_path": best_model_path,
    }




def test(model, checkpoint_path, flags, device):
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at: {checkpoint_path}"
        )

    check_point = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(check_point["model"])
    print(f"Loaded test model from: {checkpoint_path}")

    with torch.no_grad():
            val_mae, val_mape, val_rmse = evaluate(
                model, [flags.val_file], flags, device
            )
    
    print("=" * 60)
    print("VALIDATION RESULTS:")
    print(f"MAE : {val_mae:.5f}")
    print(f"MAPE: {val_mape:.5f}")
    print(f"RMSE: {val_rmse:.5f}")
    print("=" * 60)

    with torch.no_grad():
        test_mae, test_mape, test_rmse = evaluate(
            model, [flags.test_file], flags, device
        )

    print("=" * 60)
    print("TEST RESULTS:")
    print(f"MAE : {test_mae:.5f}")
    print(f"MAPE: {test_mape:.5f}")
    print(f"RMSE: {test_rmse:.5f}")
    print("=" * 60)

    return {"test_mae": test_mae, "test_mape": test_mape, "test_rmse": test_rmse, "val_mae": val_mae, "val_mape": val_mape, "val_rmse": val_rmse}


def run_experiment(args_list=None, **kwargs):
    """Entry point to run training or testing programmatically.

    Args:
        args_list (list): Optional CLI string arguments (e.g. ['--epochs', '50']).
        **kwargs: Overrides for configuration flags (e.g., train_file="data.csv", lr=0.001).

    Returns:
        dict: Test metrics {'mae': ..., 'mape': ..., 'rmse': ...}
    """
    parser = get_parser()
    flags = parser.parse_args(args_list if args_list is not None else [])

    # Override flags with any passed keyword arguments
    for key, value in kwargs.items():
        if hasattr(flags, key):
            setattr(flags, key, value)
        else:
            raise AttributeError(f"Unknown argument: {key}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flags.device = device

    # Read data info safely
    data_info_path = "data-info/data_info.json"
    if os.path.exists(data_info_path):
        with open(data_info_path, "r") as f:
            data_info = json.load(f)
    else:
        data_info = {}

    # Setup logger and hijack sys.stdout
    original_stdout = sys.stdout
    logger = logger_tb(
        flags.log_dir, flags.description, flags.code_backup, flags.use_tb
    )
    
    
    sys.stdout = message_logger(logger.log_dir)

    results = {}
    try:
        model = HierETA.HierETA_Net(flags, data_info)
        model.to(device)

        if flags.is_training:
            optimizer = optim.Adam(
                model.parameters(), lr=flags.lr, weight_decay=1e-5
            )
            results = train(model, optimizer, flags, device, logger)
        else:
            if not flags.checkpoint_path:
                raise ValueError(
                    "In test mode (`--is_training False`), you must provide a valid `checkpoint_path`."
                )
            """
            val_mae, val_mape, val_rmse = evaluate(model, eval_set, flags, device)
            print("Best Validation RESULTS:")
            print(f"MAE : {val_mae:.5f}")
            print(f"MAPE: {val_mape:.5f}")
            print(f"RMSE: {val_rmse:.5f}")
            """
            """
            val_results = {"mae": val_mae, "mape": val_mape, "rmse": val_rmse,}
            """
            
            results = test(model, flags.checkpoint_path, flags, device)
            
            
            """
            print("Test RESULTS:")
            print(f"MAE : {test_results['mae']:.5f}")
            print(f"MAPE: {test_results['mape']:.5f}")
            print(f"RMSE: {test_results['rmse']:.5f}")
            """
    finally:
        # Restore sys.stdout so subsequent function calls don't conflict
        sys.stdout = original_stdout

    return results


if __name__ == "__main__":
    # Allows running directly from terminal: python hier_eta_runner.py --epochs 20
    results = run_experiment(sys.argv[1:])
    print("Execution Finished! Returned Results:")