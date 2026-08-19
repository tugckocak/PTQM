import torch
import torch.nn as nn


class DisturbanceEncoder(nn.Module):
    """CNN encoder for the PESQM disturbance map."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 4)),
        )
        self.proj = nn.Sequential(
            nn.Linear(64 * 6 * 4, out_dim),
            nn.ReLU(),
        )

    def forward(self, dmap):
        z = self.net(dmap)
        return self.proj(z.flatten(1))


class PTQM(nn.Module):
    """Perceptual Talking Quality Model.

    Inputs
    ------
    extra_feat : Tensor, shape (B, 23)
        Acoustic descriptors and precomputed PESQM/GCC features.
    dmap : Tensor, shape (B, 1, T, Bark)
        PESQM disturbance map.

    Outputs
    -------
    Tensor, shape (B, 2)
        [m, d], where
        m = (IOS + DOS) / 2
        d = IOS - DOS
    """

    def __init__(self, extra_dim: int = 23, dmap_dim: int = 128):
        super().__init__()

        self.dmap_enc = DisturbanceEncoder(dmap_dim)

        self.trunk = nn.Sequential(
            nn.Linear(extra_dim + dmap_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
        )

        self.head_m = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.head_d = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, extra_feat, dmap):
        dmap_feat = self.dmap_enc(dmap)
        h = self.trunk(torch.cat([extra_feat, dmap_feat], dim=-1))
        return torch.cat([self.head_m(h), self.head_d(h)], dim=-1)

    @staticmethod
    def md_to_iosdos(out):
        m, d = out[:, 0], out[:, 1]
        ios = m + d / 2.0
        dos = m - d / 2.0
        return ios, dos
