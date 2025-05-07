import yaml
import argparse
import os
from datetime import datetime, timedelta
import torch
import json
import multiprocessing as mp
from training.train_mlp_core import train_mlp
from training.dataset_mlp import MLPDataset
from training.model_mlp import MLP
from training.memory_check import print_memory_usage

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_file_pairs(pattern, start_dt, end_dt, step_hours=1):
    pairs = []
    current = start_dt
    while current + timedelta(hours=step_hours) <= end_dt:
        f_t = current.strftime(pattern)
        f_tp1 = (current + timedelta(hours=step_hours)).strftime(pattern)
        if os.path.exists(f_t) and os.path.exists(f_tp1):
            pairs.append((f_t, f_tp1))
        current += timedelta(hours=step_hours)
    return pairs

def sample_random_days(start_date, end_date, num_days, seed=None):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days

    if seed is not None:
        torch.manual_seed(int(seed))

    if num_days > total_days:
        offsets = torch.randint(0, total_days, (num_days,)).tolist()
    else:
        offsets = torch.randperm(total_days)[:num_days].tolist()

    print("days sampled:", offsets)

    return [start + timedelta(days=offset) for offset in offsets]

def preload_dataset(file_pairs, norm_path, spatial_feature_path, radius, sample_ratio, seed, max_workers=4):
    print(f"Preloading {len(file_pairs)} file pairs...")
    dataset = MLPDataset(
        file_pairs=file_pairs,
        norm_path=norm_path,
        spatial_feature_path=spatial_feature_path,
        radius=radius,
        sample_ratio=sample_ratio,
        seed=seed,
        max_workers=max_workers
    )
    print(f"Preloaded {len(dataset)} samples.")
    # Tensor conversion here instead of stacking inside MLPDataset
    inputs = torch.tensor(dataset.inputs, dtype=torch.float32)
    outputs = torch.tensor(dataset.outputs, dtype=torch.float32)
    return inputs, outputs

def main(args):
    config = load_config(args.config)
    ds_cfg = config['dataset']
    train_cfg = config['training']
    random_seed = config.get('random_seed', None)

    train_set = ds_cfg['train_set']
    val_set = ds_cfg['val_set']
    resume = train_cfg.get("resume_from_checkpoint", False)
    checkpoint_path = train_cfg.get("checkpoint_path", None)

    if checkpoint_path and not resume and os.path.exists(checkpoint_path):
        print(f"Removing existing checkpoint at {checkpoint_path} since resume_from_checkpoint is False")
        os.remove(checkpoint_path)

    # === Validation ===
    print("\n[INFO] Loading validation set...")
    val_file_pairs = build_file_pairs(
        ds_cfg['path_pattern'],
        datetime.strptime(val_set['start_date'], "%Y-%m-%d"),
        datetime.strptime(val_set['end_date'], "%Y-%m-%d"),
        step_hours=1
    )
    val_inputs, val_outputs = preload_dataset(
        file_pairs=val_file_pairs,
        norm_path=ds_cfg['norm_file'],
        spatial_feature_path=ds_cfg['spatial_feature_file'],
        radius=ds_cfg['radius'],
        sample_ratio=val_set['sample_ratio'],
        seed=random_seed,
        max_workers=train_cfg.get('num_workers', 4)
    )
    print_memory_usage(val_inputs, val_outputs)

    input_dim = val_inputs.shape[1]
    output_dim = val_outputs.shape[1]
    model = MLP(input_dim, output_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_val_loss = float('inf')

    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[INFO] Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        model.eval()
        with torch.no_grad():
            pred = model(val_inputs)
            val_loss_check = torch.nn.functional.mse_loss(pred, val_outputs, reduction='mean').item()
        print(f"[INFO] Validation loss from checkpoint: {val_loss_check:.6f}")
    else:
        print("[INFO] Starting from scratch — no checkpoint loaded.")

    history = []

    # === Training Loop (1 day at a time) ===
    days = sample_random_days(
        train_set['start_date'],
        train_set['end_date'],
        train_set['num_random_days'],
        seed=random_seed
    )

    for day in days:
        print(f"\n[INFO] Processing day: {day.strftime('%Y-%m-%d')}")
        start_dt = datetime(day.year, day.month, day.day, 0)
        end_dt = start_dt + timedelta(hours=24)
        file_pairs = build_file_pairs(ds_cfg['path_pattern'], start_dt, end_dt, step_hours=1)

        if not file_pairs:
            print(f"[WARNING] No valid file pairs for {day.strftime('%Y-%m-%d')}, skipping.")
            continue

        train_inputs, train_outputs = preload_dataset(
            file_pairs=file_pairs,
            norm_path=ds_cfg['norm_file'],
            spatial_feature_path=ds_cfg['spatial_feature_file'],
            radius=ds_cfg['radius'],
            sample_ratio=train_set['sample_ratio'],
            seed=random_seed
        )
        print_memory_usage(train_inputs, train_outputs)

        train_loss, val_loss, best_val_loss = train_mlp(
            model, (train_inputs, train_outputs), (val_inputs, val_outputs), train_cfg,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_loss=best_val_loss
        )

        if checkpoint_path:
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "epoch": 0
            }, checkpoint_path)

        history.append({
            "day": day.strftime("%Y%m%d"),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss or 0.0)
        })
        with open("training_history.json", "w") as f:
            json.dump(history, f, indent=2)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args)