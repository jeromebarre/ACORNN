import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
import psutil

from training.vertical_sampling import geos_cf_sampling_weights

class MLPDataset(Dataset):
    def __init__(self, file_pairs, norm_path, radius=1, sample_ratio=0.01, spatial_feature_path=None, seed=None, max_workers=4):
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

        if spatial_feature_path is None:
            raise ValueError("You must specify spatial feature file")

        sf = xr.open_dataset(spatial_feature_path)["spatial_features"]
        self.valid_mask = ~np.isnan(sf[0, 0].values)

        self.inputs = []
        self.outputs = []

        self._parallel_generate(file_pairs, max_workers)

    def _parallel_generate(self, file_pairs, max_workers):
        rng = np.random.default_rng(self.seed)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._generate_samples_for_pair, pair, rng.integers(1e9)) for pair in file_pairs]
            for future in as_completed(futures):
                inputs, outputs = future.result()
                self.inputs.extend(inputs)
                self.outputs.extend(outputs)

    def _generate_samples_for_pair(self, pair, seed):
        f_t, f_tp1 = pair
        t0 = time.perf_counter()
        ds_t = xr.open_dataset(f_t).load()
        ds_tp1 = xr.open_dataset(f_tp1).load()
        t1 = time.perf_counter()
        print(f"[PROFILE] Loaded {os.path.basename(f_t)} and {os.path.basename(f_tp1)} in {t1 - t0:.2f}s")
        print(f"[MEMORY] After file load - RSS: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")

        rng = np.random.default_rng(seed)
        r = self.radius
        levs = ds_t[self.vars_conc[0]].shape[1]
        lat = ds_t[self.vars_conc[0]].shape[2]
        lon = ds_t[self.vars_conc[0]].shape[3]

        valid_levels = np.arange(r, levs - r)
        weights = geos_cf_sampling_weights()[valid_levels]
        weights /= weights.sum()

        total_points = (levs - 2 * r) * (lat - 2 * r) * (lon - 2 * r)
        num_samples = int(total_points * self.sample_ratio)

        lv = rng.choice(valid_levels, size=num_samples, p=weights)
        iv = rng.integers(r, lat - r, size=num_samples)
        jv = rng.integers(r, lon - r, size=num_samples)

        t2 = time.perf_counter()
        print(f"[PROFILE] Generated indices in {t2 - t1:.2f}s")

        t_sample_start = time.perf_counter()

        def extract_patches(var, ds_t, ds_tp1):
            patches = np.empty((2, num_samples, 2*r+1, 2*r+1), dtype=np.float32)
            for k, ds in enumerate([ds_t, ds_tp1]):
                data = ds[var].values[0]
                for n in range(num_samples):
                    i, j = iv[n], jv[n]
                    patches[k, n] = data[i - r:i + r + 1, j - r:j + r + 1]
            return patches

        def extract_cubes(var, ds_t, ds_tp1=None):
            n_d = 2 if ds_tp1 is not None else 1
            cubes = np.empty((n_d, num_samples, 2*r+1, 2*r+1, 2*r+1), dtype=np.float32)
            sources = [ds_t, ds_tp1] if ds_tp1 else [ds_t]
            for k, ds in enumerate(sources):
                data = ds[var].values[0]
                for n in range(num_samples):
                    l, i, j = lv[n], iv[n], jv[n]
                    cubes[k, n] = data[l - r:l + r + 1, i - r:i + r + 1, j - r:j + r + 1]
            return cubes

        def normalize(var, array):
            if var in self.log_features:
                array = np.log(array + 1e-15)
            mean = self.norm_params["means"].get(var, 0.0)
            std = self.norm_params["stds"].get(var, 1.0)
            return (array - mean) / std

        def normalize_tendency(var, val):
            mean = self.norm_params["tendency_means"].get(var, 0.0)
            std = self.norm_params["tendency_stds"].get(var, 1.0)
            return (val - mean) / std

        all_inputs = []
        for var in self.vars_2d:
            patches = extract_patches(var, ds_t, ds_tp1)
            norm_patches = [normalize(var, patches[k]) for k in range(2)]
            merged = np.stack(norm_patches, axis=1).reshape(num_samples, -1)
            all_inputs.append(merged)

        for var in self.vars_3d:
            cubes = extract_cubes(var, ds_t, ds_tp1)
            norm_cubes = [normalize(var, cubes[k]) for k in range(2)]
            merged = np.stack(norm_cubes, axis=1).reshape(num_samples, -1)
            all_inputs.append(merged)

        for var in self.vars_conc:
            cubes = extract_cubes(var, ds_t)
            norm_cube = normalize(var, cubes[0])
            merged = norm_cube.reshape(num_samples, -1)
            all_inputs.append(merged)

        inputs = np.concatenate(all_inputs, axis=1)

        outputs = []
        for n in range(num_samples):
            vec = []
            for var in self.vars_conc:
                v0 = ds_t[var].values[0, lv[n], iv[n], jv[n]]
                v1 = ds_tp1[var].values[0, lv[n], iv[n], jv[n]]
                vec.append(normalize_tendency(var + "_tend", v1 - v0))
            outputs.append(vec)

        ds_t.close()
        ds_tp1.close()

        t_sample_end = time.perf_counter()
        print(f"[PROFILE] Sampling {num_samples} points took {t_sample_end - t_sample_start:.2f}s")
        print(f"[MEMORY] After sampling {os.path.basename(f_t)} - RSS: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")

        return inputs.tolist(), outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32),
            torch.tensor(self.outputs[idx], dtype=torch.float32)
        )
