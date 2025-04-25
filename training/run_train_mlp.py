# File: ACORNN/training/run_train_mlp.py

import yaml
import argparse
from datetime import datetime, timedelta
import os
from torch.utils.data import DataLoader, random_split
from train_mlp_core import train_mlp
from dataset_mlp import MLPDataset
from model_mlp import MLP


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_file_pairs(path_pattern, start_str, end_str, step_hours):
    start = datetime.strptime(start_str, "%Y-%m-%d %H:%M")
    end = datetime.strptime(end_str, "%Y-%m-%d %H:%M")
    delta = timedelta(hours=step_hours)

    current = start
    pairs = []
    while current + delta <= end:
        file_t = current.strftime(path_pattern)
        file_tp1 = (current + delta).strftime(path_pattern)
        if os.path.exists(file_t) and os.path.exists(file_tp1):
            pairs.append((file_t, file_tp1))
        current += delta

    return pairs

def main(args):
    config = load_config(args.config)
    ds_cfg = config['dataset']
    train_cfg = config['training']

    print("Building file pairs...")
    file_pairs = build_file_pairs(
        ds_cfg['path_pattern'],
        ds_cfg['start_date'],
        ds_cfg['end_date'],
        ds_cfg['time_step_hours']
    )
    print(f"Found {len(file_pairs)} valid file pairs.")

    print("Loading dataset...")
    dataset = MLPDataset(
        file_pairs=file_pairs,
        norm_path=ds_cfg['norm_file'],
        radius=ds_cfg.get('radius', 1),
        sample_ratio=ds_cfg.get('sample_ratio', 0.01)
    )
    print(f"Loaded dataset with {len(dataset)} samples.")


    # --- SPLIT DATASET
    val_size = int(len(dataset) * train_cfg['validation_split'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False)

    # --- BUILD MODEL
    input_dim = len(dataset[0][0])
    output_dim = len(dataset[0][1])
    model = MLP(input_dim, output_dim)

    print(f"Model input dim: {input_dim}, output dim: {output_dim}")
    
    # --- NOW CALL TRAINING
    train_mlp(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg
    )

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args)
