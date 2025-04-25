# File: ACORNN/training/model_mlp.py

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_factor=2):
        super().__init__()

        hidden_dim1 = hidden_dim_factor * input_dim
        hidden_dim2 = hidden_dim1 // 2
        hidden_dim3 = hidden_dim2 // 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(),
            nn.Linear(hidden_dim3, output_dim)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    # Quick shape check
    model = MLP(input_dim=100, output_dim=6)
    x = torch.randn(10, 100)
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)