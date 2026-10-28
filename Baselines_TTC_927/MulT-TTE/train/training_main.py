import os
import sys
import shutil
from tqdm import tqdm

import torch
from torch import optim
import numpy as np

from train.train_model import train_model
from utils.prepare import create_model, create_loss
from utils.prepare import load_datadict, load_datadoct_pre
from utils.metric import calculate_metrics
from utils.util import to_var
import time


def test_model(model, data_loader, args):
    model.eval()
    predictions = list()
    targets = list()
    inds = list()
    tqdm_loader = tqdm(data_loader)
    for step, (features, truth_data) in enumerate(tqdm_loader):
        if isinstance(features, dict) and 'inds' in features.keys():
            inds.append(features['inds'])
        features = to_var(features, args.device)
        truth_data = to_var(truth_data, args.device)

        outputs, _ = model(features, args)

        targets.append(truth_data.cpu().numpy())
        predictions.append(outputs.cpu().detach().numpy())
    pre2 = np.concatenate(predictions).squeeze()
    tar2 = np.concatenate(targets)
    if len(inds) > 0:
        print(f"test size: {len(inds)}")
        print(f"test traj ids of a batch: {inds[0]}")
        inds = np.concatenate(inds)
    else:
        inds = None
        
    # Calculate metrics
    metric = calculate_metrics(pre2, tar2, args, plot=True, inds=inds)
    print(metric)
    with open(f'{args.absPath}/data/result_{args.model}.txt', 'a') as f:
        f.write(time.strftime("%m/%d %H:%M:%S",time.localtime(time.time())))
        f.write(f"epoch:{args.epochs} lr:{args.lr}\ndataset:{args.dataset} identify:{args.identify}\nloss:{args.loss}\n")
        f.write(f"{args.model_config}\n")
        f.write(f"{args.data_config}\n")
        f.write(f"{metric}\n\n")

    np.save(os.path.join(args.model_folder, "result.npy"), np.asarray([pre2, inds]))
    
    return metric


def train_main(args):
    if args.model == 'None':
        print('No chosen model')
        sys.exit(0)
    print(f"{args.mode} {args.model}_{args.identify} on {args.dataset}")
    load_datadoct_pre(args)
    data_loaders, scaler = load_datadict(args)

    args.scaler = scaler
    model = create_model(args)
    loss_func = create_loss(args)

    model_folder = f'{args.absPath}/data/save_models/{args.model}__{args.dataset}/{args.beta}_{args.lr}_{args.mask_rate}'
    args.model_folder = model_folder
    model = model.to(args.device)
    print(f'loss function: {args.loss}')
    print(f'model config: {args.model_config}')
    print(f'data config: {args.data_config}')
    print(f'arg: {args}')
    print(model)
    # 训练
    if args.mode == 'train':
        if os.path.exists(model_folder):
            shutil.rmtree(model_folder, ignore_errors=True)
        os.makedirs(model_folder, exist_ok=True)

        if args.optim == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        else:
            raise NotImplementedError()

        train_model(model=model, data_loaders=data_loaders,
                    loss_func=loss_func, optimizer=optimizer,
                    model_folder=model_folder, args=args)
    elif args.mode == 'resume':
        if args.optim == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError()
        final_model = torch.load(os.path.join(model_folder, 'final_model.pkl'), map_location=args.device)
        start_epoch = final_model['epoch']
        model.load_state_dict(final_model['model_state_dict'], strict=False)
        optimizer.load_state_dict(final_model['optimizer_state_dict'])
        train_model(model=model, data_loaders=data_loaders,
                    loss_func=loss_func, optimizer=optimizer,
                    model_folder=model_folder, args=args, start_epoch=start_epoch)
    print(args.loss)
    print(args.model_config)
    print(args.data_config)
    print(model_folder)
    
    final_metrics = {}
    best_checkpoint = torch.load(os.path.join(model_folder, 'best_model.pkl'), map_location=args.device)
    best_epoch = best_checkpoint['epoch']
    model.load_state_dict(best_checkpoint['model_state_dict'], strict=False)
    #model.load_state_dict(torch.load(os.path.join(model_folder, 'best_model.pkl'), map_location=args.device)['model_state_dict'], strict=False)
    metrics_val = test_model(model, data_loaders['val'], args)
    final_metrics['VAL_MAE'] = metrics_val.get('MAE')
    final_metrics['VAL_MAPE'] = metrics_val.get('MAPE')
    final_metrics['VAL_RMSE'] = metrics_val.get('RMSE')
    final_metrics['VAL_PEARR'] = metrics_val.get('PEARR')
    final_metrics['VAL_PEARP'] = metrics_val.get('PEARP')
        
    model.load_state_dict(
        torch.load(os.path.join(model_folder, 'best_model.pkl'), map_location=args.device)['model_state_dict'], 
        strict=False
    )
    metrics_test = test_model(model, data_loaders['test'], args)
    final_metrics['TEST_MAE'] = metrics_test.get('MAE')
    final_metrics['TEST_MAPE'] = metrics_test.get('MAPE')
    final_metrics['TEST_RMSE'] = metrics_test.get('RMSE')
    final_metrics['TEST_PEARR'] = metrics_test.get('PEARR')
    final_metrics['TEST_PEARP'] = metrics_test.get('PEARP')
    
    result_path = os.path.join(args.model_folder, "test_val_results.txt")
    
    with open(result_path, "a") as f:
        # Line 1: Epoch
        f.write(f"Epoch: {best_epoch}\n")
        
        # Line 2: Validation metrics
        f.write(f"Validation -> MAE: {final_metrics['VAL_MAE']:.4f}, MAPE: {final_metrics['VAL_MAPE']:.4f}, RMSE: {final_metrics['VAL_RMSE']:.4f}\n")
        
        # Line 3: Test metrics
        f.write(f"Test       -> MAE: {final_metrics['TEST_MAE']:.4f}, MAPE: {final_metrics['TEST_MAPE']:.4f}, RMSE: {final_metrics['TEST_RMSE']:.4f}\n")
    
    
    
    
    
    
    
    #return {'MAE': mae, 'MAPE': mape, 'RMSE': rmse, 'PEARR':pearsonrs[0], 'PEARP': pearsonrs[1]}
    
    # ADD THIS PRINT TO VERIFY:
    print(f"DEBUG IN training_main.py: metrics is {final_metrics}")
    return final_metrics

