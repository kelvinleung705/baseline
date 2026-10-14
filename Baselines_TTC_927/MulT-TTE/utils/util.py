import os
import torch


class StandardScaler2:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def save_model(path: str, **save_dict):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    torch.save(save_dict, path)


def to_var(var, device="cpu"):
    if torch.is_tensor(var):
        return var.to(device)

    if isinstance(var, (int, float, str)):
        return var

    if isinstance(var, dict):
        return {k: to_var(v, device) for k, v in var.items()}

    if isinstance(var, list):
        return [to_var(x, device) for x in var]

    if isinstance(var, tuple):
        return tuple(to_var(x, device) for x in var)

    return var