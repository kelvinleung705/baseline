import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn import SmoothL1Loss, MSELoss
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from utils.util import StandardScaler2
from models.MulT_TTE import MulT_TTE


highway = {'living_street':1, 'morotway':2, 'motorway_link':3, 'plannned':4, 'trunk':5, "secondary":6, "trunk_link":7, "tertiary_link":8, "primary":9, "residential":10, "primary_link":11, "unclassified":12, "tertiary":13, "secondary_link":14}
node_type = {'turning_circle':1, 'traffic_signals':2, 'crossing':3, 'motorway_junction':4, "mini_roundabout":5}
highway_code = {"1": 2, "2": 2, "3": 2, "4": 2, "5": 3, "6": 9, "7": 9, "8": 9, "9": 9}

list_of_pins = [[43.660334, -79.570932, 416, 1], # list_of_pins_all_dense
                [43.653276, -79.567456, 416, 1],
                [43.6462028, -79.5641287, 416, 1],
                [43.6389891, -79.5611320, 416, 1],
                [43.6316240, -79.5592392, 416, 3],
                [43.6288480, -79.555360, 230, 2],
                [43.6303126, -79.5498273, 230, 2],
                [43.6321097, -79.5446250, 230, 2],
                [43.6350280, -79.5409790, 230, 2],
                [43.6384573, -79.5379020, 230, 2]]

road_distance = [None, 0.835, 0.830, 0.838, 0.835, 0.755, 0.470, 0.468, 0.443, 0.453]


# mlm任务的输入link index中需要预测的值不能是本身，否则产生信息泄露，TTE_edge_new_data_end2end_pre更正为TTE_edge_new_data_end2end
def MulT_TTE_collate_func(data, args, info_all):
    """
    Collate function merging static road network info directly with CSV dynamic features.
    """
    linkids = []
    dateinfo = []
    inds = []
    start_segs = []
    seg_features_list = []
    times = []

    for ind, l in enumerate(data):
        start_seg = int(l[55])  # Col 55: Start segment ID (1 to 9)
        start_segs.append(start_seg)

        # Active segment IDs: from start_seg to 9 (trips end at segment 9)
        active_links = np.arange(start_seg, 10, dtype=np.int16)
        linkids.append(active_links)

        # Global temporal context (Cols 0-6: 7 features)
        g_context = l[0:7]
        dateinfo.append(g_context)

        # Compute Target Travel Time: Sum of times for active segments (Cols 9..17)
        valid_seg_times = l[9:18][start_seg - 1: 9]
        total_travel_time = np.sum(valid_seg_times)
        times.append(total_travel_time)

        inds.append(ind)

        # Build 18-dim feature vector for each active segment
        seg_feats = []
        cum_length = 0.0  # Cumulative distance counter

        for seg_id in active_links:
            # --- 1. Static Road Features ---
            hw_type = highway_code[str(seg_id)]
            seg_len = road_distance[seg_id]

            start_pin = list_of_pins[seg_id - 1]
            end_pin = list_of_pins[seg_id]
            gps_coords = [start_pin[0], start_pin[1], end_pin[0], end_pin[1]]

            static_feats = [hw_type, seg_len, cum_length] + gps_coords
            cum_length += seg_len  # Update cumulative length for next segment

            # --- 2. Dynamic Features from CSV (Cols 19..54) ---
            start_col = 19 + 4 * (seg_id - 1)
            end_col = start_col + 4
            dynamic_feats = l[start_col:end_col].tolist()

            # --- 3. Combine All Features: Static (7) + Global Context (7) + Dynamic (4) = 18 dims ---
            combined_feat = static_feats + g_context.tolist() + dynamic_feats
            seg_feats.append(combined_feat)

        seg_features_list.append(np.asarray(seg_feats, dtype=np.float32))

    # Convert targets to Tensor
    time_tensor = torch.FloatTensor(times)
    lens = np.asarray([len(k) for k in linkids], dtype=np.int16)

    # Padding mask for batching
    max_len = lens.max()
    mask = np.arange(max_len) < lens[:, None]

    # Feature dimension per segment = 18
    feat_dim = 18
    padded = np.zeros((*mask.shape, feat_dim), dtype=np.float32)

    con_links = np.concatenate(seg_features_list)
    padded[mask] = con_links

    # Raw link IDs for embeddings (padded with pad_token_id)
    pad_token_id = args.data_config['edges'] + 1
    rawlinks = np.full(mask.shape, fill_value=pad_token_id, dtype=np.int16)
    rawlinks[mask] = np.concatenate(linkids)

    # Random Masking (MLM Pre-training Task for MulT-TTE)
    def random_mask(tokens: np.ndarray, rate: float):
        replaces = np.where(np.random.random(len(tokens)) <= rate)[0]
        labels = np.full(len(tokens), dtype=np.int16, fill_value=-100)
        tokens_copy = tokens.copy()

        labels[replaces] = tokens_copy[replaces]
        tokens_copy[replaces] = pad_token_id
        return labels, tokens_copy

    mask_label_tmp = []
    sub_input_tmp = []
    for k in linkids:
        tmp1, tmp2 = random_mask(k, rate=args.mask_rate)
        mask_label_tmp.append(tmp1)
        sub_input_tmp.append(tmp2)

    mask_label = np.full(mask.shape, dtype=np.int16, fill_value=-100)
    mask_label[mask] = np.concatenate(mask_label_tmp)

    linkindex = np.full(mask.shape, fill_value=pad_token_id, dtype=np.int16)
    linkindex[mask] = np.concatenate(sub_input_tmp)

    mask_encoder = np.zeros(mask.shape, dtype=np.int16)
    mask_encoder[mask] = np.concatenate([[1] * k for k in lens])

    return {
        'links': torch.FloatTensor(padded),
        'lens': torch.LongTensor(lens),
        'inds': inds,
        'start_segs': torch.LongTensor(start_segs),
        'mask_label': torch.LongTensor(mask_label),
        'linkindex': torch.LongTensor(linkindex),
        'rawlinks': torch.LongTensor(rawlinks),
        'encoder_attention_mask': torch.LongTensor(mask_encoder)
    }, time_tensor


class BatchSampler:
    def __init__(self, dataset, batch_size):
        self.count = len(dataset)
        self.batch_size = batch_size

        if isinstance(dataset[0], np.ndarray):
            # Compute active trajectory length: (9 - start_seg + 1)
            self.lengths = [int(10 - d[55]) for d in dataset]
        elif isinstance(dataset[0], dict):
            self.lengths = [len(d['lats']) for d in dataset]
        else:
            self.lengths = [d[0]['lens'] for d in dataset]

        self.indices = list(range(self.count))

    def __iter__(self):
        np.random.shuffle(self.indices)
        chunk_size = self.batch_size * 100
        chunks = (self.count + chunk_size - 1) // chunk_size

        for i in range(chunks):
            partial_indices = self.indices[i * chunk_size: (i + 1) * chunk_size]
            partial_indices.sort(key=lambda x: self.lengths[x], reverse=True)
            self.indices[i * chunk_size: (i + 1) * chunk_size] = partial_indices

        batches = (self.count - 1 + self.batch_size) // self.batch_size
        for i in range(batches):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size


def load_datadoct_pre(args):
    global info_all
    abspath = os.path.join(os.path.dirname(__file__), "data_config.json")
    with open(abspath) as file:
        data_config = json.load(file)[args.dataset]
        args.data_config = data_config

    # All static and dynamic features are handled directly in Python / CSV
    info_all = [None, None, None, None]


class Datadict(Dataset):
    def __init__(self, inputs):
        self.content = inputs

    def __getitem__(self, idx):
        return self.content[idx]

    def __len__(self):
        return len(self.content)


def load_datadict(args):
    data = {}
    loader = {}
    phases = ['test'] if args.mode == 'test' else ['train', 'val', 'test']

    for phase in phases:
        csv_path = Path(args.absPath) / args.data_config['data_dir'] / f"{phase}.csv"
        df = pd.read_csv(csv_path)
        data[phase] = df.to_numpy(dtype=np.float32)
        print(f"Loaded {phase}.csv with shape: {data[phase].shape}")

        loader[phase] = DataLoader(
            Datadict(data[phase]),
            batch_sampler=BatchSampler(data[phase], args.data_config['batch_size']),
            collate_fn=lambda x: eval(args.data_config['collate_fn'])(x, args, info_all),
            pin_memory=True
        )

    loader['test'] = DataLoader(
        Datadict(data['test']),
        batch_size=args.data_config['batch_size'],
        collate_fn=lambda x: eval(args.data_config['collate_fn'])(x, args, info_all),
        shuffle=False,
        pin_memory=True
    )

    return loader.copy(), StandardScaler2(
        mean=args.data_config['time_mean'],
        std=args.data_config['time_std']
    )


def create_model(args):
    absPath = os.path.join(os.path.dirname(__file__), "model_config.json")
    with open(absPath) as file:
        model_config = json.load(file)[args.model]
    args.model_config = model_config
    model_config['pad_token_id'] = args.data_config['edges'] + 1
    if "MulT_TTE" in args.model:
        return MulT_TTE(**model_config)


def create_loss(args):
    if args.loss == 'rmse':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            return torch.sqrt(torch.mean(torch.pow(preds - labels, 2)))
    elif args.loss == 'mse':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            return MSELoss(reduction='mean').forward(preds.view(-1), labels)
    elif args.loss == 'mape':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            return torch.mean(torch.abs(preds - labels) / (labels + 0.1))
    elif args.loss == 'mae':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            return torch.mean(torch.abs(preds - labels))
    elif args.loss == 'smoothL1':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            preds = torch.squeeze(preds, 1)
            return SmoothL1Loss(reduction='mean', beta=args.loss_val).forward(preds, labels)
    else:
        raise ValueError("Unknown loss function.")
    return loss