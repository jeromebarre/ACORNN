import os
import json
from datetime import datetime, timedelta
import xarray as xr
import torch
from training.model_mlp import MLP
from forecast.fc_step import apply_forecast_step


def build_forecast_pairs(path_pattern, start_time, forecast_hours):
    pairs = []
    for h in range(forecast_hours):
        t0 = start_time + timedelta(hours=h)
        t1 = t0 + timedelta(hours=1)
        f0 = t0.strftime(path_pattern)
        f1 = t1.strftime(path_pattern)
        if not os.path.exists(f0) or not os.path.exists(f1):
            raise FileNotFoundError(f"Missing file: {f0} or {f1}")
        pairs.append((f0, f1))
    return pairs


def write_forecast_step(ds_tmpl, pred_conc, output_path):
    ds_out = ds_tmpl.copy(deep=True)
    for var in pred_conc:
        ds_out[var].values[0] = pred_conc[var][0]  # shape: [1, lev, lat, lon]
    ds_out.to_netcdf(output_path)
    print(f"[INFO] Wrote forecast step to {output_path}")


def main():
    # === Configuration ===
    path_pattern = "/Users/jeromebarre/AI/GEOS_CF_US/GEOS_CF_%Y%m%d_%H00z.nc4"
    norm_path = "/Users/jeromebarre/AI/ACORNN/moments/normalization_params_all.json"
    model_ckpt = "/Users/jeromebarre/AI/ACORNN/checkpoints/GEOS_CF_US_mlp_latest_save.pth"
    output_dir = "forecast_outputs"
    os.makedirs(output_dir, exist_ok=True)

    forecast_start = datetime(2024, 1, 30, 0)
    forecast_hours = 6
    radius = 1
    constrain_edges = True
    num_workers = 4
    batch_size = 10000
    device = "cpu"  # or "mps" if supported and faster

    # === Load model ===
    ckpt = torch.load(model_ckpt, map_location=device)
    input_dim = ckpt['model_state']['net.0.weight'].shape[1]
    output_dim = list(ckpt['model_state'].values())[-1].shape[0]
    model = MLP(input_dim, output_dim)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    # === Load normalization params ===
    with open(norm_path) as f:
        norm_params = json.load(f)

    # === Build forecast file pairs ===
    file_pairs = build_forecast_pairs(path_pattern, forecast_start, forecast_hours)

    # === Initial concentration ===
    ds0 = xr.open_dataset(file_pairs[0][0])
    pred_conc = {var: ds0[var].copy(deep=True) for var in ["CO", "HCHO", "NO", "NO2", "O3", "SO2"]}

    # === Recursive Forecast ===
    for step, (f_t, f_tp1) in enumerate(file_pairs):
        print(f"[INFO] Forecast step {step}: {os.path.basename(f_t)} -> {os.path.basename(f_tp1)}")
        ds_t = xr.open_dataset(f_t).load()
        ds_tp1 = xr.open_dataset(f_tp1).load()

        pred_conc = apply_forecast_step(
            ds_t, ds_tp1, pred_conc, model,
            norm_params=norm_params,
            radius=radius,
            device=device,
            constrain_edges=constrain_edges,
            num_workers=num_workers,
            batch_size=batch_size
        )

        output_path = os.path.join(
            output_dir, f"forecast_{forecast_start + timedelta(hours=step+1):%Y%m%d_%H}z.nc"
        )
        write_forecast_step(ds_tp1, pred_conc, output_path)


if __name__ == "__main__":
    main()