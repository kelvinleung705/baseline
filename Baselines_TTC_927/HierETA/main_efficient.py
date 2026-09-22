import os
import sys
import json
import argparse
import torch
import torch.optim as optim

import utils
import dataloading
from models import HierETA
from log import logger_tb, message_logger

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--is_training', type=bool, default=True, help="training mode or not")

parser.add_argument('--segment_num', type=int, default=4, help="segment number per link")
parser.add_argument('--link_num', type=int, default=3, help="link number per route")

parser.add_argument('--win_size', type=int, default=3, help="window scale of neighboring segments")
parser.add_argument('--Lambda', type=float, default=0.4, help="weighting parameter in decoder")
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")

parser.add_argument('--data_dir', type=str, default="./samples/", help="directory for route data storage")
parser.add_argument('--train_file', type=str, default="train_trips.csv", help="training csv filename")
parser.add_argument('--eval_file', type=str, default="test_trips.csv", help="evaluation csv filename")

parser.add_argument('--log_dir', type=str, default="logs")
parser.add_argument('--use_tb', type=bool, default=False, help='Use tensorboard to log training info')
parser.add_argument('--code_backup', type=bool, default=True, help='code backup or not')
parser.add_argument('--description', type=str, default="HierETA", help='description of current running experiments.')

FLAGS = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FLAGS.device = device

# Load data_info
data_info_path = 'data-info/data_info_old.json'
if os.path.exists(data_info_path):
    with open(data_info_path, 'r') as f:
        data_info = json.load(f)
else:
    data_info = {}

logger = logger_tb(FLAGS.log_dir, FLAGS.description, FLAGS.code_backup, FLAGS.use_tb)
sys.stdout = message_logger(logger.log_dir)


def train(model, optimizer, scheduler):
    train_set = [FLAGS.train_file]
    eval_set = [FLAGS.eval_file]
    model.to(device)

    best_eval_mae = float('inf')
    step = 0

    for epoch in range(FLAGS.epochs):
        model.train()
        print(f"\n--- Training epoch {epoch} ---")

        for input_file in train_set:
            data_iter = dataloading.get_loader(input_file, FLAGS)
            data_iter_len = len(data_iter)

            for idx, attr in enumerate(data_iter):
                attr = utils.to_var(attr, device)

                optimizer.zero_grad()
                pred, label = model(attr)
                loss = utils.MAE(pred, label)
                loss.backward()

                # Gradient clipping to prevent gradient explosions in LSTMs
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                optimizer.step()
                step += 1

                if (idx + 1) % 50 == 0:
                    mape = utils.MAPE(pred, label)
                    rmse = utils.RMSE(pred, label)
                    print(f"Epoch: [{epoch}/{FLAGS.epochs}] Step: {step} "
                          f"Progress: {(idx + 1) * 100 / data_iter_len:.2f}% | "
                          f"MAE: {loss.item():.2f} MAPE: {mape.item():.4f} RMSE: {rmse.item():.2f}")

        # Evaluate at the end of each epoch
        eval_mape, eval_mae, eval_rmse = evaluate(model, eval_set)
        
        # Step LR scheduler based on validation MAE
        scheduler.step(eval_mae)

        # Save ONLY the best checkpoint
        if eval_mae < best_eval_mae:
            best_eval_mae = eval_mae
            best_weight_name = f"best_model_epoch-{epoch}.pth"
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "step": step,
                "best_mae": best_eval_mae
            }
            save_path = os.path.join(logger.log_dir, best_weight_name)
            torch.save(checkpoint, save_path)
            print(f">>> [BEST] Model saved with MAE: {best_eval_mae:.2f} at {save_path}")


def evaluate(model, files):
    model.eval()
    MAE_loss, MAPE_loss, RMSE_loss = [], [], []

    with torch.no_grad():
        for input_file in files:
            data_iter = dataloading.get_loader(input_file, FLAGS)
            for attr in data_iter:
                attr = utils.to_var(attr, device)
                pred, label = model(attr)

                MAE_loss.append(utils.MAE(pred, label).item())
                MAPE_loss.append(utils.MAPE(pred, label).item())
                RMSE_loss.append(utils.RMSE(pred, label).item())

    mean_mape = sum(MAPE_loss) / len(MAPE_loss)
    mean_mae = sum(MAE_loss) / len(MAE_loss)
    mean_rmse = sum(RMSE_loss) / len(RMSE_loss)

    print("\n" + "=" * 40)
    print(f"Validation Summary:\nMAE:  {mean_mae:.4f}\nMAPE: {mean_mape:.4f}\nRMSE: {mean_rmse:.4f}")
    print("=" * 40 + "\n")

    return mean_mape, mean_mae, mean_rmse


def test(model_path):
    model = HierETA.HierETA_Net(FLAGS, data_info)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    print(f"Loaded test model from {model_path}")

    eval_set = [FLAGS.eval_file]
    evaluate(model, eval_set)


if __name__ == '__main__':
    model = HierETA.HierETA_Net(FLAGS, data_info)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=1e-5)
    
    # Decays LR when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    if FLAGS.is_training:
        train(model, optimizer, scheduler)
    else:
        best_checkpoint = "logs/.../best_model_epoch-X.pth"
        test(best_checkpoint)