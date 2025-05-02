import yaml
import argparse
import os
from datetime import datetime, timedelta
import torch
from torch.utils.data import DataLoader
import json

from train_mlp_core import train_mlp
from dataset_mlp import MLPDataset
from model_mlp import MLP

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
    selected_offsets = torch.randperm(total_days)[:num_days].tolist()
    return [start + timedelta(days=offset) for offset in selected_offsets]

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

    val_file_pairs = build_file_pairs(
        ds_cfg['path_pattern'],
        datetime.strptime(val_set['start_date'], "%Y-%m-%d"),
        datetime.strptime(val_set['end_date'], "%Y-%m-%d"),
        step_hours=1
    )
    print(f"\nProcessing validation set with {len(val_file_pairs)} file pairs")
    #print(val_file_pairs)
    val_dataset = MLPDataset(
        file_pairs=val_file_pairs,
        norm_path=ds_cfg['norm_file'],
        spatial_feature_path=ds_cfg['spatial_feature_file'],
        radius=ds_cfg['radius'],
        sample_ratio=val_set['sample_ratio'],
        seed=random_seed
    )
    print("MLPDataset Done")
    
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=train_cfg.get('num_workers', 0))
    print("Torch DataLoader Done")
    
    input_dim = len(val_dataset[0][0])
    output_dim = len(val_dataset[0][1])
    model = MLP(input_dim, output_dim)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_val_loss = float('inf')
    
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        model.eval()
        val_loss_check = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to("cpu"), yb.to("cpu")
                pred = model(xb)
                loss = torch.nn.functional.mse_loss(pred, yb, reduction='sum')
                val_loss_check += loss.item()
        val_loss_check /= len(val_loader.dataset)
        print(f"Validation loss from checkpoint state: {val_loss_check:.6f}")      
    else:
        print("Starting from scratch — no checkpoint loaded.")
    
    history = []

    days = sample_random_days(
        train_set['start_date'],
        train_set['end_date'],
        train_set['num_random_days'],
        seed=random_seed
    )

    for day in days:
        start_dt = datetime(day.year, day.month, day.day, 0)
        end_dt = start_dt + timedelta(hours=24)

        file_pairs = build_file_pairs(ds_cfg['path_pattern'], start_dt, end_dt, step_hours=1)
        if not file_pairs:
            print(f"No valid file pairs for {day.strftime('%Y-%m-%d')}, skipping")
            continue

        print(f"\nProcessing day {day.strftime('%Y-%m-%d')} with {len(file_pairs)} file pairs")
        #print(file_pairs)
        dataset = MLPDataset(
            file_pairs=file_pairs,
            norm_path=ds_cfg['norm_file'],
            spatial_feature_path=ds_cfg['spatial_feature_file'],
            radius=ds_cfg['radius'],
            sample_ratio=train_set['sample_ratio'],
            seed=random_seed
        )
        print("MLPDataset Done")
        
        train_loader = DataLoader(dataset, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=train_cfg.get('num_workers', 0))
        print("Torch DataLoader Done")

        print(f"Training samples for {day}: {len(dataset)}")
        train_loss, val_loss, best_val_loss = train_mlp(
            model, train_loader, val_loader, train_cfg,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_loss=best_val_loss
        )

        #if val_loss < best_val_loss - float(train_cfg.get('stop_delta', 1e-4)):
            #print(f"New best val loss: {val_loss:.6f} (prev {best_val_loss:.6f}), saving checkpoint.")
            #best_val_loss = val_loss
        if checkpoint_path:
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "epoch": 0  # optional: update if epoch-tracking is restored
                }, checkpoint_path)
                
        day_str = day.strftime("%Y%m%d")
        history.append({"day": day_str, "train_loss": float(train_loss), "val_loss": float(val_loss or 0.0)})
        with open("training_history.json", "w") as f:
            json.dump(history, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args)
