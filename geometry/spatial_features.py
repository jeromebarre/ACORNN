# File: ACORNN/geometry/spatial_features.py

import yaml
import numpy as np
import xarray as xr
import argparse
from geom_functions import SpatialGeometry

def parse_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(config):
    # --- Load config ---
    sample_file = config["dataset"]["sample_file"]
    radius = config["spatial_features"]["radius"]
    save_path = config["spatial_features"]["save_path"]

    DEGREE_RAD = np.pi / 180  # Normalization factor
    
    # --- Load lat/lon grid ---
    ds = xr.open_dataset(sample_file)
    lat = ds["lat"].values
    lon = ds["lon"].values
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")

    n_lat, n_lon = lat2d.shape
    n_neighbors = (2 * radius + 1)**2 - 1  # exclude center cell
    features = np.full((n_neighbors, 2, n_lat, n_lon), np.nan, dtype=np.float32)

    # --- Neighbor offsets (excluding center) ---
    offsets = []
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            if di == 0 and dj == 0:
                continue
            offsets.append((di, dj))

    # --- Compute features for all valid cells ---
    for idx, (di, dj) in enumerate(offsets):
        for i in range(radius, n_lat - radius):
            for j in range(radius, n_lon - radius):
                lat_c = lat2d[i, j]
                lon_c = lon2d[i, j]
                lat_n = lat2d[i + di, j + dj]
                lon_n = lon2d[i + di, j + dj]

                dist = SpatialGeometry.haversine_distance(lat_c, lon_c, lat_n, lon_n)
                azim = SpatialGeometry.azimuth_angle(lat_c, lon_c, lat_n, lon_n)

                # Normalize angular distance by 1°
                features[idx, 0, i, j] = dist / DEGREE_RAD
                features[idx, 1, i, j] = azim

    # --- Save to NetCDF ---
    out_ds = xr.Dataset(
        {
            "spatial_features": (["neighbor", "feature", "lat", "lon"], features)
        },
        coords={
            "neighbor": np.arange(n_neighbors),
            "feature": ["angular_distance", "azimuth_angle"],
            "lat": lat,
            "lon": lon
        }
    )
    out_ds.to_netcdf(save_path)
    print(f" Saved spatial feature file to {save_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = parse_config(args.config)
    main(config)
