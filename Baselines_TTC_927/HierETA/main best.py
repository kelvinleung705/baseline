import argparse
import json
import os
import sys
from random import shuffle
import torch
import torch.optim as optim

# -------------------------------------------------------------------
# ADD THIS LINE TO FIX CUDNN_STATUS_BAD_PARAM:
torch.backends.cudnn.enabled = False
# -------------------------------------------------------------------

import dataloading
from log import logger_tb, message_logger
from models import HierETA
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument(
    "--is_training", type=bool, default=True, help="training mode or not"
)

parser.add_argument(
    "--segment_num", type=int, default=4, help="segment number per link"
)
parser.add_argument(
    "--link_num", type=int, default=3, help="link number per route"
)

parser.add_argument(
    "--win_size",
    type=int,
    default=3,
    help="window scale of neighboring segments",
)
parser.add_argument(
    "--Lambda", type=float, default=0.4, help="weighting parameter in decoder"
)
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")

# Dataset directory and CSV file names
parser.add_argument(
    "--data_dir",
    type=str,
    default="./samples/",
    help="directory for route data storage",
)
parser.add_argument(
    "--train_file",
    type=str,
    default="train_trips.csv",
    help="training csv filename",
)
parser.add_argument(
    "--eval_file",
    type=str,
    default="validation_trips.csv",
    help="evaluation csv filename",
)
parser.add_argument(
    "--test_file", type=str, default="test_trips.csv", help="test csv filename"
)  # Added test file

parser.add_argument("--log_dir", type=str, default="logs")
parser.add_argument(
    "--step_per_eval",
    type=int,
    default=100,
    help="training steps per evaluation",
)
parser.add_argument(
    "--use_tb",
    type=bool,
    default=False,
    help="Use tensorboard to log training info",
)
parser.add_argument(
    "--code_backup", type=bool, default=True, help="code backup or not"
)
parser.add_argument(
    "--description",
    type=str,
    default="HierETA",
    help="description of current running experiments.",
)

FLAGS = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLAGS.device = device

# Load or fallback data_info
data_info_path = "data-info/data_info.json"
if os.path.exists(data_info_path):
    data_info = json.load(open(data_info_path, "r"))
else:
    data_info = {}  # Fallback if json is not present

# code backup and message logging
logger = logger_tb(
    FLAGS.log_dir, FLAGS.description, FLAGS.code_backup, FLAGS.use_tb
)
sys.stdout = message_logger(logger.log_dir)


def train(model, optimizer):
    train_set = [FLAGS.train_file]
    eval_set = [FLAGS.eval_file]
    test_set = [FLAGS.test_file]

    print("train file nums: ", len(train_set))
    model.train()
    model.to(device)

    step = 0
    best_mae = float("inf")
    best_model_path = os.path.join(logger.log_dir, "best_model.pth")

    for epoch in range(FLAGS.epochs):
        print("train files " + str(train_set))
        print("eval files " + str(eval_set))
        print("--- Training epoch {} ---".format(epoch))

        for input_file in train_set:
            model.train()
            print("--- Train on file {} ---".format(input_file))

            data_iter = dataloading.get_loader(input_file, FLAGS)
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
                            loss,
                            mape,
                            rmse,
                        )
                    )

                # Periodic validation
                if step and step % FLAGS.step_per_eval == 0:
                    with torch.no_grad():
                        val_mae, val_mape, val_rmse = evaluate(model, eval_set)

                        # Save checkpoint only if validation MAE has improved
                        if val_mae < best_mae:
                            best_mae = val_mae
                            check_point = {
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "epoch": epoch,
                                "step": step,
                                "best_mae": best_mae,
                            }
                            torch.save(check_point, best_model_path)
                            print(
                                ">>> Best model saved to {} with Val MAE: {:.5f} <<<\n".format(
                                    best_model_path, best_mae
                                )
                            )

                    model.train()

    # =========================================================================
    # After training finishes: Load best model and evaluate on test_trips.csv
    # =========================================================================
    print(
        "\n" + "=" * 60 + "\n TRAINING FINISHED. RUNNING EVALUATION ON TEST SET\n"
    )
    if os.path.exists(best_model_path):
        print(f"Loading best checkpoint from: {best_model_path}")
        check_point = torch.load(best_model_path, map_location=device)
        model.load_state_dict(check_point["model"])
    else:
        print(
            "Warning: No checkpoint was saved during training. Testing with current weights."
        )

    print("Evaluating Best Model on:", test_set)
    with torch.no_grad():
        test_mae, test_mape, test_rmse = evaluate(model, test_set)

    print("=" * 60)
    print("FINAL TEST RESULTS (test_trips):")
    print("Test MAE : {:.5f}".format(test_mae))
    print("Test MAPE: {:.5f}".format(test_mape))
    print("Test RMSE: {:.5f}".format(test_rmse))
    print("=" * 60 + "\n")


def evaluate(model, files):
    model.eval()
    MAE_loss = []
    MAPE_loss = []
    RMSE_loss = []

    for file_idx, input_file in enumerate(files):
        MAE_loss_single_file = []
        MAPE_loss_single_file = []
        RMSE_loss_single_file = []

        data_iter = dataloading.get_loader(input_file, FLAGS)
        for idx, attr in enumerate(data_iter):
            attr = utils.to_var(attr, device)
            pred, label = model(attr)

            mae = utils.MAE(pred, label)
            mape = utils.MAPE(pred, label)
            rmse = utils.RMSE(pred, label)

            MAE_loss_single_file.append(mae.item())
            RMSE_loss_single_file.append(rmse.item())
            MAPE_loss_single_file.append(mape.item())

            if idx > 10 and idx % 100 == 0:
                print(
                    "Evaluate Progress: {:.5f}".format(
                        (idx + 1) * 100 / len(data_iter)
                    )
                )
                print(
                    "step: {}, MAE_loss {:.5f}".format(
                        idx,
                        sum(MAE_loss_single_file) / len(MAE_loss_single_file),
                    )
                )
                print(
                    "step: {}, MAPE_loss {:.5f}".format(
                        idx,
                        sum(MAPE_loss_single_file) / len(MAPE_loss_single_file),
                    )
                )
                print(
                    "step: {}, RMSE_loss {:.5f}".format(
                        idx,
                        sum(RMSE_loss_single_file) / len(RMSE_loss_single_file),
                    )
                )

        print("***********************")
        print(
            "Evaluate on file {}, MAE_loss {:.5f}".format(
                input_file,
                sum(MAE_loss_single_file) / len(MAE_loss_single_file),
            )
        )
        print(
            "Evaluate on file {}, MAPE_loss {:.5f}".format(
                input_file,
                sum(MAPE_loss_single_file) / len(MAPE_loss_single_file),
            )
        )
        print(
            "Evaluate on file {}, RMSE_loss {:.5f}".format(
                input_file,
                sum(RMSE_loss_single_file) / len(RMSE_loss_single_file),
            )
        )
        print("***********************\n\n")
        MAE_loss.extend(MAE_loss_single_file)
        MAPE_loss.extend(MAPE_loss_single_file)
        RMSE_loss.extend(RMSE_loss_single_file)

    MAPE = sum(MAPE_loss) / len(MAPE_loss) if MAPE_loss else 0
    MAE = sum(MAE_loss) / len(MAE_loss) if MAE_loss else float("inf")
    RMSE = sum(RMSE_loss) / len(RMSE_loss) if RMSE_loss else 0

    print(
        "\n--------final----------- \nMAPE: {:.5f} \nMAE:{:.5f} \nRMSE:{:.5f}\n".format(
            MAPE, MAE, RMSE
        )
    )

    return MAE, MAPE, RMSE


def test(model_path=""):
    model = HierETA.HierETA_Net(FLAGS, data_info)
    file = os.path.join(model_path, "best_model.pth")
    check_point = torch.load(file, map_location=device)
    model.load_state_dict(check_point["model"])
    print("Loaded test model: " + file)
    model.to(device)

    # Evaluates directly on test_file
    with torch.no_grad():
        test_mae, test_mape, test_rmse = evaluate(model, [FLAGS.test_file])

    print("=" * 60)
    print("TEST RESULTS:")
    print("MAE : {:.5f}".format(test_mae))
    print("MAPE: {:.5f}".format(test_mape))
    print("RMSE: {:.5f}".format(test_rmse))
    print("=" * 60)


if __name__ == "__main__":
    model = HierETA.HierETA_Net(FLAGS, data_info)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=1e-5)

    if FLAGS.is_training:
        train(model, optimizer)
    else:
        model_path = "logs/2021-12-25-10-35-07_JustForDemo"
        test(model_path)