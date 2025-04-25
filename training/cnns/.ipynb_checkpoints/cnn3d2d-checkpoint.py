import torch
import torch.nn as nn

class CNN3D2D(nn.Module):
    def __init__(self, in_channels_3d, in_channels_2d, in_channels_emis, out_channels):
        super().__init__()

        self.encoder_3d = nn.Sequential(
            nn.Conv3d(in_channels_3d, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.encoder_2d = nn.Sequential(
            nn.Conv2d(in_channels_2d, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.encoder_emis = nn.Sequential(
            nn.Conv2d(in_channels_emis, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Conv3d(64 + 16 + 16, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, out_channels, kernel_size=1)
        )

    def forward(self, x3d, x2d, emis):
        # x3d: (B, C3, L, H, W)
        # x2d, emis: (B, C2 or Ce, H, W)
        B, _, L, H, W = x3d.shape

        x3d = self.encoder_3d(x3d)  # -> (B, 64, L, H, W)

        x2d = self.encoder_2d(x2d)  # -> (B, 16, H, W)
        x2d = x2d.unsqueeze(2).expand(-1, -1, L, -1, -1)  # broadcast to (B, 16, L, H, W)

        emis = self.encoder_emis(emis)  # -> (B, 16, H, W)
        emis_exp = torch.zeros((B, 16, L, H, W), device=emis.device)
        emis_exp[:, :, -10:, :, :] = emis.unsqueeze(2)  # inject into last 10 levels (surface)

        fused = torch.cat([x3d, x2d, emis_exp], dim=1)  # (B, 96, L, H, W)
        out = self.fusion(fused)  # (B, out_channels, L, H, W)
        return out

