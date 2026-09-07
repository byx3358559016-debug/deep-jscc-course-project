"""Lightweight Deep JSCC models and differentiable channel layers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.activation = nn.PReLU(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.body(x))


class PowerNormalize(nn.Module):
    """Normalize every sample to unit average real-symbol power."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        power = x.square().mean(dim=(1, 2, 3), keepdim=True)
        return x / torch.sqrt(power + self.eps)


def _noise_for_snr(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    signal_power = x.square().mean(dim=(1, 2, 3), keepdim=True).detach()
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    noise_std = torch.sqrt(signal_power / snr_linear)
    return torch.randn_like(x) * noise_std


def awgn_channel(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    return x + _noise_for_snr(x, snr_db)


def slow_rayleigh_channel(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    """One real Rayleigh fading gain per image, without explicit CSI."""

    shape = (x.shape[0], 1, 1, 1)
    real = torch.randn(shape, device=x.device, dtype=x.dtype)
    imag = torch.randn(shape, device=x.device, dtype=x.dtype)
    gain = torch.sqrt(real.square() + imag.square()) / math.sqrt(2.0)
    faded = gain * x
    return faded + _noise_for_snr(x, snr_db)


class SemanticCommSystem(nn.Module):
    def __init__(
        self,
        latent_channels: int = 16,
        residual_blocks: int = 0,
        power_normalization: bool = False,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.residual_blocks = residual_blocks
        self.power_normalization = power_normalization
        improved = residual_blocks > 0 or power_normalization

        def activation(channels: int) -> nn.Module:
            return nn.PReLU(channels) if improved else nn.ReLU()

        encoder_layers: list[nn.Module] = [
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            activation(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            activation(64),
            nn.Conv2d(64, latent_channels, 3, padding=1),
        ]
        encoder_layers.extend(ResidualBlock(latent_channels) for _ in range(residual_blocks))
        if power_normalization:
            encoder_layers.append(PowerNormalize())
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        decoder_layers.extend(ResidualBlock(latent_channels) for _ in range(residual_blocks))
        decoder_layers.extend(
            [
                nn.ConvTranspose2d(latent_channels, 64, 3, padding=1),
                activation(64),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                activation(32),
                nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
                nn.Sigmoid(),
            ]
        )
        self.decoder = nn.Sequential(*decoder_layers)

    def channel_forward(self, x: torch.Tensor, snr_db: float, channel: str = "awgn") -> torch.Tensor:
        if channel == "awgn":
            return awgn_channel(x, snr_db)
        if channel == "rayleigh":
            return slow_rayleigh_channel(x, snr_db)
        if channel == "clean":
            return x
        raise ValueError(f"unknown channel: {channel}")

    def forward(self, x: torch.Tensor, snr_db: float, channel: str = "awgn") -> torch.Tensor:
        encoded = self.encoder(x)
        received = self.channel_forward(encoded, snr_db, channel)
        return self.decoder(received)


def build_model(config: dict) -> SemanticCommSystem:
    return SemanticCommSystem(
        latent_channels=int(config["latent_channels"]),
        residual_blocks=int(config.get("residual_blocks", 0)),
        power_normalization=bool(config.get("power_normalization", False)),
    )
