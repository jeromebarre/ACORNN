import torch
import torch.nn as nn
import time

# Dummy MLP similar to your project
class DummyMLP(nn.Module):
    def __init__(self, input_dim=512, hidden=1024, output_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Benchmark function
def benchmark(device, n_batches=100, batch_size=256, input_dim=512):
    model = DummyMLP(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    x = torch.randn(batch_size, input_dim).to(device)
    y = torch.randn(batch_size, 6).to(device)

    torch.cuda.empty_cache() if device.type == "cuda" else None

    # Warm-up
    for _ in range(10):
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Timing
    torch.mps.synchronize() if device.type == "mps" else None
    start = time.time()
    for _ in range(n_batches):
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.mps.synchronize() if device.type == "mps" else None
    end = time.time()

    print(f"{device}: {(end - start):.2f} s for {n_batches} batches")

# Run
if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)

    if torch.backends.mps.is_available():
        benchmark(torch.device("mps"))
    else:
        print("MPS not available on this system.")

    benchmark(torch.device("cpu"))