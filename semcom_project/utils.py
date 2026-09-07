"""Shared reproducibility, metrics and path utilities."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
DEFAULT_DATA_DIR = ROOT_DIR / "basic_code" / "kodak"
DEFAULT_SPLIT_FILE = PROJECT_DIR / "configs" / "split.json"
OUTPUT_DIR = PROJECT_DIR / "outputs"


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def bandwidth_ratio(latent_channels: int) -> float:
    """k/n when two real latent values form one complex channel symbol."""

    return latent_channels / 96.0


def compute_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float]:
    reconstructed = reconstructed.detach().float().clamp(0, 1).unsqueeze(0)
    original = original.detach().float().to(reconstructed.device).unsqueeze(0)
    mse = functional.mse_loss(reconstructed, original)
    psnr = 10.0 * torch.log10(1.0 / mse)

    channels = original.shape[1]
    coordinates = torch.arange(11, dtype=original.dtype, device=original.device) - 5
    gaussian = torch.exp(-(coordinates.square()) / (2.0 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    window = torch.outer(gaussian, gaussian).expand(channels, 1, 11, 11)
    mu_x = functional.conv2d(original, window, padding=5, groups=channels)
    mu_y = functional.conv2d(reconstructed, window, padding=5, groups=channels)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x_sq = functional.conv2d(original.square(), window, padding=5, groups=channels) - mu_x_sq
    sigma_y_sq = functional.conv2d(reconstructed.square(), window, padding=5, groups=channels) - mu_y_sq
    sigma_xy = functional.conv2d(original * reconstructed, window, padding=5, groups=channels) - mu_xy
    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    )
    return {
        "mse": float(mse.cpu().item()),
        "psnr": float(psnr.cpu().item()),
        "ssim": float(ssim_map.mean().cpu().item()),
    }


def save_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
