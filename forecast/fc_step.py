import torch
import numpy as np
import time
import psutil
from concurrent.futures import ThreadPoolExecutor

def apply_forecast_step(
    ds_t, ds_tp1, pred_conc, model, norm_params,
    radius=1, device="cpu", constrain_edges=False, num_workers=4, batch_size=10000
):
    log_features = set(norm_params.get("log_features", []))
    vars_2d = ["PS", "CLDTT", "SZA", "EMIS_SO2", "EMIS_NO", "EMIS_HCHO", "EMIS_CO"]
    vars_3d = ["Q", "QCTOT", "T", "U", "V", "logP"]
    vars_conc = ["CO", "HCHO", "NO", "NO2", "O3", "SO2"]

    def normalize(var, data):
        if var in log_features:
            data = np.log(data + 1e-15)
        mean = norm_params["means"].get(var, 0.0)
        std = norm_params["stds"].get(var, 1.0)
        return (data - mean) / std

    def unnormalize_tendency(var, data):
        mean = norm_params["tendency_means"].get(var, 0.0)
        std = norm_params["tendency_stds"].get(var, 1.0)
        return data * std + mean

    r = radius
    levs, lat, lon = ds_t["CO"].shape[1:]
    t_start = time.perf_counter()

    l_idx, i_idx, j_idx = np.meshgrid(
        np.arange(r, levs - r),
        np.arange(r, lat - r),
        np.arange(r, lon - r),
        indexing='ij'
    )
    flat_indices = list(zip(
        l_idx.flatten(), i_idx.flatten(), j_idx.flatten()
    ))

    def process_chunk(chunk_indices):
        inputs_chunk = []
        l_chunk, i_chunk, j_chunk = zip(*chunk_indices)

        def extract_patch_2d(ds, var):
            data = ds[var].values[0]
            return np.stack([
                data[i - r:i + r + 1, j - r:j + r + 1].reshape(-1)
                for i, j in zip(i_chunk, j_chunk)
            ])

        def extract_cube_3d(ds, var):
            data = ds[var].values[0]
            return np.stack([
                data[l - r:l + r + 1, i - r:i + r + 1, j - r:j + r + 1].reshape(-1)
                for l, i, j in zip(l_chunk, i_chunk, j_chunk)
            ])

        def extract_conc(var):
            data = pred_conc[var].values[0]
            return np.stack([
                data[l - r:l + r + 1, i - r:i + r + 1, j - r:j + r + 1].reshape(-1)
                for l, i, j in zip(l_chunk, i_chunk, j_chunk)
            ])

        for var in vars_2d:
            t0 = time.perf_counter()
            inputs_chunk.append(normalize(var, extract_patch_2d(ds_t, var)))
            inputs_chunk.append(normalize(var, extract_patch_2d(ds_tp1, var)))
            print(f"[PROFILE] 2D var {var} processed in chunk in {time.perf_counter() - t0:.2f}s")

        for var in vars_3d:
            t0 = time.perf_counter()
            inputs_chunk.append(normalize(var, extract_cube_3d(ds_t, var)))
            inputs_chunk.append(normalize(var, extract_cube_3d(ds_tp1, var)))
            print(f"[PROFILE] 3D var {var} processed in chunk in {time.perf_counter() - t0:.2f}s")

        for var in vars_conc:
            t0 = time.perf_counter()
            inputs_chunk.append(normalize(var, extract_conc(var)))
            print(f"[PROFILE] Concentration var {var} processed in chunk in {time.perf_counter() - t0:.2f}s")

        input_np = np.concatenate(inputs_chunk, axis=1)

        est_mem_MB = input_np.nbytes / 1024**2
        total_mem_MB = psutil.virtual_memory().available / 1024**2
        print(f"[MEMORY] Estimated input size: {est_mem_MB:.2f} MB ({(est_mem_MB / total_mem_MB)*100:.1f}% of available)")

        # === Batched inference
        pred_chunk = []
        with torch.no_grad():
            for b in range(0, len(input_np), batch_size):
                batch = torch.tensor(input_np[b:b + batch_size], dtype=torch.float32, device=device)
                out = model(batch).cpu().numpy()
                pred_chunk.append(out)
        pred_chunk = np.concatenate(pred_chunk, axis=0)

        updates = {var: np.zeros_like(pred_conc[var].values[0]) for var in vars_conc}
        for i, var in enumerate(vars_conc):
            delta = unnormalize_tendency(var + "_tend", pred_chunk[:, i])
            updates[var][l_chunk, i_chunk, j_chunk] = delta
        return updates

    chunk_size = len(flat_indices) // num_workers
    chunks = [flat_indices[i:i + chunk_size] for i in range(0, len(flat_indices), chunk_size)]
    all_updates = {var: np.zeros_like(pred_conc[var].values[0]) for var in vars_conc}

    print(f"[INFO] Dispatching {len(chunks)} chunks to {num_workers} threads...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        for fut in futures:
            result = fut.result()
            for var in vars_conc:
                all_updates[var] += result[var]

    for var in vars_conc:
        pred_conc[var].values[0] += all_updates[var]
        #clamping
        pred_conc[var].values[0] = np.maximum(pred_conc[var].values[0], 1e-18)
        if constrain_edges:
            pred_conc[var].values[0, :, :r, :] = ds_tp1[var].values[0, :, :r, :]
            pred_conc[var].values[0, :, -r:, :] = ds_tp1[var].values[0, :, -r:, :]
            pred_conc[var].values[0, :, :, :r] = ds_tp1[var].values[0, :, :, :r]
            pred_conc[var].values[0, :, :, -r:] = ds_tp1[var].values[0, :, :, -r:]

    print(f"[PROFILE] Total forecast step time: {time.perf_counter() - t_start:.2f}s")
    print(f"[MEMORY] Final memory usage - RSS: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")

    return pred_conc