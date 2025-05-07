import torch
from torch.utils.data import DataLoader

class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, input_dim, output_dim):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randn(self.input_dim)
        y = torch.randn(self.output_dim)
        return x, y

if __name__ == "__main__":
    # Parameters for synthetic dataset
    num_samples = 10000
    input_dim = 128
    output_dim = 1
    batch_size = 16

    # Create synthetic dataset and DataLoader
    train_dataset = SyntheticDataset(num_samples, input_dim, output_dim)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    print(f"Testing DataLoader with {num_samples} samples, batch size {batch_size}...")

    # Measure loading time for each batch
    import time
    start_time = time.time()
    for i, (xb, yb) in enumerate(train_loader):
        batch_time = time.time() - start_time
        print(f"Batch {i + 1} loaded in {batch_time:.4f} seconds, x shape: {xb.shape}, y shape: {yb.shape}")
        start_time = time.time()

        # Stop after a few batches for testing
        if i == 5:
            break