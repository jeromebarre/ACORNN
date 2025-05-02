import os
from dataset_mlp import MLPDataset

# File pairs: GEOS-CF t and t+1
file_pairs = [
    (
        "/discover/nobackup/projects/jcsda/s2127/barre/GEOS_CF_US/output/GEOS_CF_20240101_0000z.nc4",
        "/discover/nobackup/projects/jcsda/s2127/barre/GEOS_CF_US/output/GEOS_CF_20240101_0100z.nc4"
    )
]

# Normalization and spatial feature paths
norm_path = "/discover/nobackup/jbarre/jedidev/ACORNN/moments/normalization_params_all.json"
spatial_feature_path = "/discover/nobackup/jbarre/jedidev/ACORNN/geometry/spatial_features_GEOS_CF_US.nc"

# Instantiate the dataset with spatial features
dataset = MLPDataset(
    file_pairs=file_pairs,
    norm_path=norm_path,
    spatial_feature_path=spatial_feature_path,
    radius=1,
    sample_ratio=0.001
)

print(f"Total samples: {len(dataset)}")
x, y = dataset[0]
print("Input shape:", x.shape)
print("Output shape (tendencies):", y.shape)

# Print input values in chunks of 9 (to visualize 3×3 structure)
print("Input values grouped by 9:")
for i in range(0, len(x), 9):
    chunk = x[i:i+9].numpy()
    print(f"{i:03d}-{i+8:03d}: {chunk}")

print("Output values (tendencies):", y.numpy())

# Test resampling
print("\nResampling dataset...")
dataset.resample()

print(f"Total samples after resample: {len(dataset)}")
x, y = dataset[0]
print("Input shape after resample:", x.shape)
print("Output shape after resample:", y.shape)

print("Input values grouped by 9 (after resample):")
for i in range(0, len(x), 9):
    chunk = x[i:i+9].numpy()
    print(f"{i:03d}-{i+8:03d}: {chunk}")

print("Output values (tendencies after resample):", y.numpy())
