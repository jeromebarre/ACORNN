import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_mlp(model, train_data, val_data, config, optimizer, scheduler, best_val_loss):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    train_inputs, train_outputs = [t.to(device) for t in train_data]
    val_inputs, val_outputs = [t.to(device) for t in val_data]

    loss_fn = nn.MSELoss()
    epochs_without_improvement = 0
    batch_size = config.get("batch_size", None) or len(train_inputs)

    for epoch in range(config["max_epochs"]):
        model.train()
        train_loss = 0.0

        for start in range(0, len(train_inputs), batch_size):
            end = start + batch_size
            xb = train_inputs[start:end]
            yb = train_outputs[start:end]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_inputs)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for start in range(0, len(val_inputs), batch_size):
                end = start + batch_size
                xb = val_inputs[start:end]
                yb = val_outputs[start:end]
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_inputs)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss - float(config.get('stop_delta', 1e-4)):
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["patience"]:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    return train_loss, val_loss, best_val_loss