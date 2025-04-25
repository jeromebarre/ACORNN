# File: ACORNN/training/datasets.py
import json
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset

class Atmospheric3DSplitDataset(Dataset):
    def __init__(self, file_list, norm_json, vars_3d, vars_2d, emis_vars, target_vars):
        with open(norm_json, 'r') as f:
            stats = json.load(f)

        self.means = stats['means']
        self.stds = stats['stds']
        self.log_features = stats['log_features']

        self.vars_3d = vars_3d
        self.vars_2d = vars_2d
        self.emis_vars = emis_vars
        self.target_vars = target_vars
        self.samples = []

        for fpath in file_list:
            ds = xr.open_dataset(fpath)

            def norm(var, arr):
                if var in self.log_features:
                    arr = np.log(arr + 1e-15)
                return (arr - self.means[var]) / self.stds[var]

            X3d = [norm(var, ds[var].squeeze().values) for var in vars_3d]
            X3d = np.stack(X3d, axis=0).astype(np.float32)  # (C3, L, H, W)

            X2d = [norm(var, ds[var].squeeze().values) for var in vars_2d]
            X2d = np.stack(X2d, axis=0).astype(np.float32)  # (C2, H, W)

            EMIS = [norm(var, ds[var].squeeze().values) for var in emis_vars]
            EMIS = np.stack(EMIS, axis=0).astype(np.float32)  # (Ce, H, W)

            Y = [ds[var].squeeze().values for var in target_vars]
            Y = np.stack(Y, axis=0).astype(np.float32)  # (Ct, L, H, W)

            self.samples.append((X3d, X2d, EMIS, Y))
            ds.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X3d, X2d, EMIS, Y = self.samples[idx]
        return (torch.tensor(X3d), torch.tensor(X2d), torch.tensor(EMIS), torch.tensor(Y))

