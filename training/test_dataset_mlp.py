# File: test_mlp_dataset.py

import os
from dataset_mlp import MLPDataset

# Replace with actual file paths
file_pairs = [
    (
        "/discover/nobackup/projects/jcsda/s2127/barre/GEOS_CF_US/output/GEOS_CF_20240101_0000z.nc4",
        "/discover/nobackup/projects/jcsda/s2127/barre/GEOS_CF_US/output/GEOS_CF_20240101_0100z.nc4"
    )
]

norm_path = "/discover/nobackup/jbarre/jedidev/ACORNN/moments/normalization_params_all.json"

# Instantiate the dataset
dataset = MLPDataset(file_pairs, norm_path, radius=1, sample_ratio=0.001)


print(f"Total samples: {len(dataset)}")
x, y = dataset[0]
print("Input shape:", x.shape)
print("Output shape (tendencies):", y.shape)

# Print input values in chunks of 9 (like 3x3 patches/slices)
print("Input values grouped by 9:")
for i in range(0, len(x), 9):
    chunk = x[i:i+9].numpy()
    print(f"{i:03d}-{i+8:03d}: {chunk}")

print("Output values (tendencies):", y.numpy())

# Now test resampling
print("\nResampling dataset...")
dataset.resample()

print(f"Total samples after resample: {len(dataset)}")
x, y = dataset[0]
print("Input shape after resample:", x.shape)
print("Output shape after resample:", y.shape)

# Print input values in chunks of 9 again after resample
print("Input values grouped by 9 (after resample):")
for i in range(0, len(x), 9):
    chunk = x[i:i+9].numpy()
    print(f"{i:03d}-{i+8:03d}: {chunk}")

print("Output values (tendencies after resample):", y.numpy())