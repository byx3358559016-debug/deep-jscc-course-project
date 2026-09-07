from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import load_split
from model import PowerNormalize, SemanticCommSystem, awgn_channel
from utils import DEFAULT_SPLIT_FILE, bandwidth_ratio


def test_split_is_disjoint_and_complete():
    split = load_split(DEFAULT_SPLIT_FILE)
    all_names = split["train"] + split["val"] + split["test"]
    assert len(all_names) == 24
    assert len(set(all_names)) == 24


def test_model_preserves_image_shape():
    model = SemanticCommSystem(latent_channels=8, residual_blocks=1, power_normalization=True)
    output = model(torch.rand(2, 3, 64, 64), snr_db=5)
    assert output.shape == (2, 3, 64, 64)
    assert float(output.detach().min()) >= 0.0
    assert float(output.detach().max()) <= 1.0


def test_power_normalization_is_per_sample():
    normalized = PowerNormalize()(torch.rand(3, 8, 16, 16) * 4)
    powers = normalized.square().mean(dim=(1, 2, 3))
    assert torch.allclose(powers, torch.ones_like(powers), atol=1e-5)


def test_awgn_has_expected_snr_statistically():
    torch.manual_seed(7)
    signal = torch.ones(16, 4, 64, 64)
    received = awgn_channel(signal, 10.0)
    empirical_snr = signal.square().mean() / (received - signal).square().mean()
    assert abs(10.0 * torch.log10(empirical_snr).item() - 10.0) < 0.25


def test_bandwidth_ratios_match_paper_convention():
    assert bandwidth_ratio(8) == 1 / 12
    assert bandwidth_ratio(16) == 1 / 6
    assert bandwidth_ratio(32) == 1 / 3
