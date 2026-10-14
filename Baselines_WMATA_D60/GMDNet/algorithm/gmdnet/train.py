import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from my_utils.utils import *
import numpy as np
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def seed_it(seed):
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)
seed_it(1024)


class GMDNet_Dataset(Dataset):
    def __init__(
            self,
            mode: str,
            params: dict,
    ) -> None:
        super().__init__()
        if mode not in ["train", "val", "test"]:
            raise ValueError
        path_key = {'train': 'train_path', 'val': 'val_path', 'test': 'test_path'}[mode]
        path = params[path_key]
        print(path)
        self.data = np.load(path, allow_pickle=True).item()

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, index):
        route = self.data['route'][index]
        mask = self.data['mask'][index]
        f = self.data['f'][index]
        label = self.data['label'][index]

        return route, mask, f, label

def collate_fn(batch):
    return batch

def process_batch(batch, model, device, edge, node, A):
    route, mask, f, label = zip(*batch)
    route = torch.LongTensor(route).to(device)
    mask = torch.LongTensor(mask).to(device)
    f = torch.FloatTensor(f).to(device)
    
    # Scale seconds to minutes for training stability (500s -> 8.33 min)
    label = torch.FloatTensor(label).to(device) / 60.0

    edge = torch.FloatTensor(edge).to(device)
    node = torch.FloatTensor(node).to(device)
    A = torch.LongTensor(A).to(device)

    negative_log_likelihood = model.forward(route, mask, f, edge, node, A, label, 'train')

    return negative_log_likelihood

def test_model(model, test_dataloader, device, edge, node, A):
    predicts_list = []
    label_list = []
    nll_list = []
    crps_list = []
    model.eval()
    edge = torch.FloatTensor(edge).to(device)
    node = torch.FloatTensor(node).to(device)
    A = torch.LongTensor(A).to(device)

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            route, mask, f, label = zip(*batch)
            route = torch.LongTensor(route).to(device)
            mask = torch.LongTensor(mask).to(device)
            f = torch.FloatTensor(f).to(device)
            # Ensure label is (B, 1) on device in seconds
            label_sec = torch.FloatTensor(label).to(device).view(-1, 1)

            # Model was trained on minutes, so feed label in minutes
            pi, mu, sigma = model.forward(route, mask, f, edge, node, A, label_sec / 60.0, 'test')

            # Convert distribution parameters to seconds
            mu_sec = mu * 60.0
            sigma_sec = sigma * 60.0

            # 1. Prediction (expected value in seconds)
            pred = torch.sum(mu_sec * pi, dim=1)  # shape: (B,)

            # 2. Correct GMM Negative Log-Likelihood in seconds
            m = torch.distributions.Normal(loc=mu_sec, scale=sigma_sec)
            log_component_prob = m.log_prob(label_sec)  # shape: (B, K)
            log_pi = torch.log(torch.clamp(pi, min=1e-8))
            # log(sum_k pi_k * N_k)
            log_likelihood = torch.logsumexp(log_pi + log_component_prob, dim=-1)  # shape: (B,)
            batch_nll = -torch.mean(log_likelihood)
            nll_list.append(batch_nll.item())  # Store scalar float

            # 3. CRPS calculation (in seconds)
            crps_metric = CRPSMetric(label_sec, mu_sec, sigma_sec)
            get_crps = crps_metric.gaussian_crps()
            crps = torch.sum(torch.mul(get_crps, pi), dim=1)
            crps_list.append(crps.mean().item())  # Store scalar float

            # Collect predictions and labels
            predicts_list.extend(pred.view(-1).cpu().tolist())
            label_list.extend(label_sec.view(-1).cpu().tolist())

        predicts = np.array(predicts_list)
        labels = np.array(label_list)
        val_nll = np.mean(nll_list)
        val_crps = np.mean(crps_list)

        from sklearn.metrics import mean_absolute_error as mae

        # 1. MAE
        val_mae = mae(labels, predicts)
        print("Labels[:10]:", labels[:10])
        print("Preds[:10]: ", predicts[:10])

        # 2. RMSE
        val_rmse = np.sqrt(np.mean((labels - predicts) ** 2))

        # 3. Safe MAPE
        valid_mask = labels > 1e-2
        val_mape = np.mean(np.abs((labels[valid_mask] - predicts[valid_mask]) / labels[valid_mask])) if np.sum(valid_mask) > 0 else 0.0

        # Print outputs
        print(f'mae: {val_mae:.4f}')
        print(f'rmse: {val_rmse:.4f}')
        print(f'mape: {val_mape:.4f} ({val_mape * 100:.2f}%)')
        print(f'negative-log-likelihood: {val_nll:.4f}')
        print(f'CRPS: {val_crps:.4f}')

        return val_mae, val_mape, val_nll, val_crps

def get_params():
    from my_utils.utils import get_common_params
    parser = get_common_params()
    # Model parameters
    parser.add_argument('--model', type=str, default='GMDNet')
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--n_gaussians', type=int, default=5)
    parser.add_argument('--is_eval', type=str, default=True, help='True means load existing model')
    parser.add_argument('--att_hidden_size', type=int, default=64, help='dims of route sa hidden dim')
    parser.add_argument('--num_of_attention_heads', type=int, default=8, help='dims of route sa hidden dim')
    parser.add_argument('--num_layers', type=int, default=2, help='num of gnn layer')
    parser.add_argument('--dirichlet_alpha', type=int, default=1, help='Dirichlet regularizer')

    # Dimensions matching your dataset
    parser.add_argument('--edge_dim', type=int, default=4, help='Number of dynamic segment features')
    parser.add_argument('--max_seq_len', type=int, default=10, help='Max number of segments in a trip')
    parser.add_argument('--node_dim', type=int, default=3, help='Node feature dimension')
    parser.add_argument('--route_fea_dim', type=int, default=8, help='Global temporal feature dimension')

    args, _ = parser.parse_known_args()
    return args

from my_utils.utils import run

def main(params):
    run(params, GMDNet_Dataset, process_batch, test_model, collate_fn)

if __name__ == '__main__':
    import time, nni, torch
    import logging

    print('lets start')
    logger = logging.getLogger('training')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise