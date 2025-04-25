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

def build_file_list(config):
    path_pattern = config['dataset']['path_pattern']
    start = datetime.strptime(config['dataset']['start_date'], "%Y-%m-%d %H:%M")
    end = datetime.strptime(config['dataset']['end_date'], "%Y-%m-%d %H:%M")
    delta = timedelta(hours=config['dataset']['time_step_hours'])

    current = start
    files = []

    while current <= end:
        file_path = current.strftime(path_pattern)
        if os.path.exists(file_path):
            files.append(file_path)
        else:
            print(f"Warning: Missing file {file_path}")
        current += delta

    return files

def save_moments(means, stds, log_features, save_path):
    params = {
        "means": means,
        "stds": stds,
        "log_features": log_features
    }
    with open(save_path, 'w') as f:
        json.dump(params, f)
    print(f"Saved normalization parameters to {save_path}")

def main(args):
    config = load_yaml_config(args.config)
    log_features = config['features']['log_transform_features']
    output_path = config['output']['save_path']

    files = build_file_list(config)

    # Open one file to list variables
    sample_ds = xr.open_dataset(files[0])
    all_vars = [v for v in sample_ds.data_vars if v not in ['time', 'lat', 'lon', 'lev']]
    sample_ds.close()

    print(f"Found {len(all_vars)} variables: {all_vars}")

    running_means = {var: 0.0 for var in all_vars}
    running_M2s = {var: 0.0 for var in all_vars}
    running_counts = {var: 0 for var in all_vars}

    for file in files:
        print(f"Processing {file}")
        ds = xr.open_dataset(file)
        for var in all_vars:
            data = ds[var].squeeze().values.flatten()
            data = data[~np.isnan(data)]  # Remove NaNs

            if var in log_features:
                data = np.log(data + 1e-15)

            if data.size == 0:
                continue  # skip if empty

            batch_size = len(data)
            batch_mean = np.mean(data)
            batch_var = np.var(data)

            delta = batch_mean - running_means[var]
            total_count = running_counts[var] + batch_size

            running_means[var] += delta * (batch_size / total_count)
            running_M2s[var] += batch_var * batch_size + (delta**2) * running_counts[var] * batch_size / total_count
            running_counts[var] = total_count 
        ds.close()

    # Finalize
    means = {}
    stds = {}
    for var in all_vars:
        if running_counts[var] > 1:
            means[var] = running_means[var]
            stds[var] = np.sqrt(running_M2s[var] / (running_counts[var] - 1))
        else:
            means[var] = np.nan
            stds[var] = np.nan
            print(f"Warning: not enough samples for {var}")

    save_moments(means, stds, log_features, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute independent moments (mean, std) for each NetCDF variable.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file.')
    args = parser.parse_args()

    main(args)

