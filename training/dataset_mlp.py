# File: ACORNN/training/dataset_mlp.py
import json
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
from pathlib import Path

class MLPDataset(Dataset):
    def __init__(self, file_pairs, norm_path, radius=1, sample_ratio=0.01):
        self.file_pairs = file_pairs
        self.radius = radius
        self.sample_ratio = sample_ratio
        self.samples = []

        with open(norm_path) as f:
            self.norm_params = json.load(f)

        self.vars_2d = ["PS", "CLDTT", "SZA", "EMIS_SO2", "EMIS_NO", "EMIS_HCHO", "EMIS_CO"]
        self.vars_3d = ["Q", "QCTOT", "T", "U", "V", "logP"]
        self.vars_conc = ["CO", "HCHO", "NO", "NO2", "O3", "SO2"]
        self.target_vars = [v + "_tend" for v in self.vars_conc]

        self.log_features = set(self.norm_params.get("log_features", []))

        self.resample()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def _normalize(self, var, data):
        if var in self.log_features:
            data = np.log(data + 1e-15)
        mean = self.norm_params["means"].get(var, 0.0)
        std = self.norm_params["stds"].get(var, 1.0)
        return (data - mean) / std

    def _normalize_tendency(self, var, data):
        mean = self.norm_params["tendency_means"].get(var, 0.0)
        std = self.norm_params["tendency_stds"].get(var, 1.0)
        return (data - mean) / std

    def _extract_patch(self, arr, i, j):
        if (i < self.radius or j < self.radius or
            i + self.radius >= arr.shape[-2] or j + self.radius >= arr.shape[-1]):
            arr_padded = np.pad(arr, self.radius, mode='edge') if arr.ndim == 2 else np.pad(arr, ((0, 0), (self.radius, self.radius), (self.radius, self.radius)), mode='edge')
            i += self.radius
            j += self.radius
        else:
            arr_padded = arr
        if arr.ndim == 2:
            return arr_padded[i:i + 2 * self.radius + 1, j:j + 2 * self.radius + 1]
        elif arr.ndim == 3:
            return arr_padded[:, i:i + 2 * self.radius + 1, j:j + 2 * self.radius + 1]
        else:
            raise ValueError(f"Unsupported array shape for patch extraction: {arr.shape}")

    def _extract_cube(self, arr, l, i, j):
        needs_pad = (
            l < self.radius or l + self.radius >= arr.shape[0] or
            i < self.radius or i + self.radius >= arr.shape[1] or
            j < self.radius or j + self.radius >= arr.shape[2]
        )
        if needs_pad:
            arr_padded = np.pad(arr, ((self.radius, self.radius), (self.radius, self.radius), (self.radius, self.radius)), mode='edge')
            l_p, i_p, j_p = l + self.radius, i + self.radius, j + self.radius
        else:
            arr_padded = arr
            l_p, i_p, j_p = l, i, j
        return arr_padded[
            l_p - self.radius:l_p + self.radius + 1,
            i_p - self.radius:i_p + self.radius + 1,
            j_p - self.radius:j_p + self.radius + 1
        ]

    def resample(self):
        self.samples = []
    
        def vertical_sampling_weights(n_levs):
            levels = np.arange(n_levs)
            weights = np.exp(-0.03 * levels) + 0.2 * np.exp(-((levels - 50) ** 2) / (2 * 5 ** 2))
            return weights / np.sum(weights)
    
        for f_t, f_tp1 in self.file_pairs:
            ds_t = xr.open_dataset(f_t)
            ds_tp1 = xr.open_dataset(f_tp1)
    
            # --- PAD ALL DATA ARRAYS ---
            def pad_ds(ds):
                padded = {}
                for var in self.vars_2d + self.vars_3d + self.vars_conc:
                    arr = ds[var].values
                    if arr.ndim == 4:  # (time, lev, lat, lon)
                        arr = np.pad(arr, ((0,0), (self.radius,self.radius), (self.radius,self.radius), (self.radius,self.radius)), mode='edge')
                    elif arr.ndim == 3:  # (time, lat, lon) for 2D vars
                        arr = np.pad(arr, ((0,0), (self.radius,self.radius), (self.radius,self.radius)), mode='edge')
                    padded[var] = arr
                return padded
    
            ds_t_padded = pad_ds(ds_t)
            ds_tp1_padded = pad_ds(ds_tp1)
    
            levs = ds_t_padded[self.vars_conc[0]].shape[1]  # already padded
            lat = ds_t_padded[self.vars_conc[0]].shape[2]
            lon = ds_t_padded[self.vars_conc[0]].shape[3]
    
            sampling_weights_full = vertical_sampling_weights(levs - 2*self.radius)
            sampling_weights = sampling_weights_full
            sampling_weights /= sampling_weights.sum()
    
            total_points = (levs - 2*self.radius) * (lat - 2*self.radius) * (lon - 2*self.radius)
            num_samples = int(total_points * self.sample_ratio)
            rng = np.random.default_rng()
            selected_points = []
    
            while len(selected_points) < num_samples:
                l = rng.choice(np.arange(self.radius, levs - self.radius), p=sampling_weights)
                i = rng.integers(self.radius, lat - self.radius)
                j = rng.integers(self.radius, lon - self.radius)
                selected_points.append((l, i, j))
    
            for l, i, j in selected_points:
                input_vec = []
    
                # 2D vars
                for var in self.vars_2d:
                    for arr in [ds_t_padded[var], ds_tp1_padded[var]]:
                        patch = arr[0, i - self.radius:i + self.radius + 1, j - self.radius:j + self.radius + 1]
                        input_vec.extend(self._normalize(var, patch).flatten())
    
                # 3D vars
                for var in self.vars_3d:
                    for arr in [ds_t_padded[var], ds_tp1_padded[var]]:
                        cube = arr[0, l - self.radius:l + self.radius + 1, i - self.radius:i + self.radius + 1, j - self.radius:j + self.radius + 1]
                        input_vec.extend(self._normalize(var, cube).flatten())
    
                # concentrations at time t
                for var in self.vars_conc:
                    arr = ds_t_padded[var]
                    cube = arr[0, l - self.radius:l + self.radius + 1, i - self.radius:i + self.radius + 1, j - self.radius:j + self.radius + 1]
                    input_vec.extend(self._normalize(var, cube).flatten())
    
                output_vec = []
                for var in self.vars_conc:
                    tend = ds_tp1_padded[var][0, l, i, j] - ds_t_padded[var][0, l, i, j]
                    output_vec.append(self._normalize_tendency(var + "_tend", tend))
    
                self.samples.append((torch.tensor(input_vec, dtype=torch.float32),
                                      torch.tensor(output_vec, dtype=torch.float32)))
    
            ds_t.close()
            ds_tp1.close()

