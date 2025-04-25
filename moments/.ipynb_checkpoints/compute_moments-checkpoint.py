import yaml
import xarray as xr
import numpy as np
import os
import argparse
from datetime import datetime, timedelta
import json

def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_file_pairs(config):
    path_pattern = config['dataset']['path_pattern']
    start = datetime.strptime(config['dataset']['start_date'], "%Y-%m-%d %H:%M")
    end = datetime.strptime(config['dataset']['end_date'], "%Y-%m-%d %H:%M")
    delta = timedelta(hours=config['dataset']['time_step_hours'])

    current = start
    pairs = []

    while current + delta <= end:
        file_t = current.strftime(path_pattern)
        file_tp1 = (current + delta).strftime(path_pattern)
        if os.path.exists(file_t) and os.path.exists(file_tp1):
            pairs.append((file_t, file_tp1))
        else:
            print(f"Warning: Skipping pair ({file_t}, {file_tp1})")
        current += delta

    return pairs

def main(args):
    config = load_yaml_config(args.config)
    log_features = config['features']['log_transform_features']
    tendency_vars = config['features']['tendency_variables']
    output_path = config['output']['save_path']

    file_pairs = build_file_pairs(config)

    # Initialize running stats for all variables
    # Collect all relevant variables from the first dataset
    sample_ds = xr.open_dataset(file_pairs[0][0])
    all_vars = [v for v in sample_ds.data_vars if v not in ['time', 'lat', 'lon', 'lev']]
    all_vars += [v for v in tendency_vars]
    print(all_vars)
    sample_ds.close()
    running_means = {var: 0.0 for var in all_vars}
    running_M2s = {var: 0.0 for var in all_vars}
    running_counts = {var: 0 for var in all_vars}

    for file_t, file_tp1 in file_pairs:
        print(f"Processing pair: {file_t}, {file_tp1}")
        ds_t = xr.open_dataset(file_t)
        ds_tp1 = xr.open_dataset(file_tp1)

        for var in all_vars:

            # Handle tendency data
            if var.endswith("_tend"):
                base_var = var.replace("_tend", "")
                data_t = ds_t[base_var].squeeze().values
                data_tp1 = ds_tp1[base_var].squeeze().values
                data = data_tp1 - data_t


            # Always update concentration stats
            if not var.endswith("_tend"):
                data = ds_t[var].squeeze().values
                if var in log_features:
                    data = np.log(data + 1e-15)

            flat = data.flatten()
            flat = flat[~np.isnan(flat)]

            if flat.size > 0:
                batch_size = flat.size
                batch_mean = np.mean(flat)
                batch_var = np.var(flat)

                delta = batch_mean - running_means[var]
                total_count = running_counts[var] + batch_size

                running_means[var] += delta * (batch_size / total_count)
                running_M2s[var] += batch_var * batch_size + (delta ** 2) * running_counts[var] * batch_size / total_count
                running_counts[var] = total_count

        ds_t.close()
        ds_tp1.close()

    # Finalize
    means = {}
    stds = {}
    print(all_vars)
    for var in all_vars:
        if running_counts[var] > 1:
            means[var] = running_means[var]
            stds[var] = np.sqrt(running_M2s[var] / (running_counts[var] - 1))
        else:
            means[var] = np.nan
            stds[var] = np.nan
            print(f"Warning: not enough samples for {var}")

    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            existing = json.load(f)
    else:
        existing = {}

    existing["means"] = {var: means[var] for var in means if not var.endswith("_tend")}
    existing["stds"] = {var: stds[var] for var in stds if not var.endswith("_tend")}
    existing["tendency_means"] = {var: means[var] for var in means if var.endswith("_tend")}
    existing["tendency_stds"] = {var: stds[var] for var in stds if var.endswith("_tend")}
    existing["log_features"] = log_features

    with open(output_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"Saved tendency moments to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute mean and stddev of concentration tendencies.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file.')
    args = parser.parse_args()

    main(args)

