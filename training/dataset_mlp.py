import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr
import json

from vertical_sampling import geos_cf_sampling_weights

class MLPDataset(Dataset):
    def __init__(self, file_pairs, norm_path, radius=1, sample_ratio=0.01, spatial_feature_path=None, seed=None):
        self.radius = radius
        self.sample_ratio = sample_ratio
        self.seed = seed

        with open(norm_path) as f:
            self.norm_params = json.load(f)

        self.vars_2d = ["PS", "CLDTT", "SZA", "EMIS_SO2", "EMIS_NO", "EMIS_HCHO", "EMIS_CO"]
        self.vars_3d = ["Q", "QCTOT", "T", "U", "V", "logP"]
        self.vars_conc = ["CO", "HCHO", "NO", "NO2", "O3", "SO2"]
        self.target_vars = [v + "_tend" for v in self.vars_conc]
        self.log_features = set(self.norm_params.get("log_features", []))

        self.file_pairs = file_pairs
        self.loaded_data = {}
        self.inputs = []
        self.outputs = []

        if spatial_feature_path is None:
            raise ValueError("You must specify spatial feature file")

        sf = xr.open_dataset(spatial_feature_path)["spatial_features"]
        self.valid_mask = ~np.isnan(sf[0, 0].values)

        self._preload_unique_files()
        self._generate_samples()

        self.loaded_data.clear()

    def _preload_unique_files(self):
        unique_files = set()
        for f_t, f_tp1 in self.file_pairs:
            unique_files.add(f_t)
            unique_files.add(f_tp1)

        for path in unique_files:
            with xr.open_dataset(path) as ds:
                self.loaded_data[path] = ds.load()

    def _generate_samples(self):
        rng = np.random.default_rng(self.seed)
        for f_t, f_tp1 in self.file_pairs:
            ds_t = self.loaded_data[f_t]
            ds_tp1 = self.loaded_data[f_tp1]

            levs = ds_t[self.vars_conc[0]].shape[1]
            lat = ds_t[self.vars_conc[0]].shape[2]
            lon = ds_t[self.vars_conc[0]].shape[3]

            valid_levels = np.arange(self.radius, levs - self.radius)
            weights = geos_cf_sampling_weights()[valid_levels]
            weights /= weights.sum()

            total_points = (levs - 2 * self.radius) * (lat - 2 * self.radius) * (lon - 2 * self.radius)
            num_samples = int(total_points * self.sample_ratio)

            for _ in range(num_samples):
                l = rng.choice(valid_levels, p=weights)
                i = rng.integers(self.radius, lat - self.radius)
                j = rng.integers(self.radius, lon - self.radius)

                input_vec = []

                for var in self.vars_2d:
                    for ds in [ds_t, ds_tp1]:
                        patch = self._extract_patch(ds[var].values[0], i, j)
                        input_vec.extend(self._normalize(var, patch).flatten())

                for var in self.vars_3d:
                    for ds in [ds_t, ds_tp1]:
                        cube = self._extract_cube(ds[var].values[0], l, i, j)
                        input_vec.extend(self._normalize(var, cube).flatten())

                for var in self.vars_conc:
                    cube = self._extract_cube(ds_t[var].values[0], l, i, j)
                    input_vec.extend(self._normalize(var, cube).flatten())

                output_vec = []
                for var in self.vars_conc:
                    v0 = ds_t[var].values[0, l, i, j]
                    v1 = ds_tp1[var].values[0, l, i, j]
                    output_vec.append(self._normalize_tendency(var + "_tend", v1 - v0))

                self.inputs.append(input_vec)
                self.outputs.append(output_vec)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32),
            torch.tensor(self.outputs[idx], dtype=torch.float32)
        )

    def _extract_patch(self, data, i, j):
        return data[i - self.radius:i + self.radius + 1, j - self.radius:j + self.radius + 1]

    def _extract_cube(self, data, l, i, j):
        return data[l - self.radius:l + self.radius + 1,
                    i - self.radius:i + self.radius + 1,
                    j - self.radius:j + self.radius + 1]

    def _normalize(self, var, data):
        if var in self.log_features:
            data = np.log(data + 1e-15)
        mean = self.norm_params["means"].get(var, 0.0)
        std = self.norm_params["stds"].get(var, 1.0)
        return (data - mean) / std

    def _normalize_tendency(self, var, val):
        mean = self.norm_params["tendency_means"].get(var, 0.0)
        std = self.norm_params["tendency_stds"].get(var, 1.0)
        return (val - mean) / std
