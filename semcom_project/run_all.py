"""Run the five core experiments and generate all evaluation artifacts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

EXPERIMENTS = [
    ("baseline_c16_random", "baseline", 16, [-10, -5, 0, 5, 10]),
    ("improved_c8_random", "improved", 8, [-10, -5, 0, 5, 10]),
    ("improved_c16_random", "improved", 16, [-10, -5, 0, 5, 10]),
    ("improved_c32_random", "improved", 32, [-10, -5, 0, 5, 10]),
    ("improved_c16_fixed5", "improved", 16, [5]),
]


def run(command: list[str]) -> None:
    print("\n>", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="retrain existing checkpoints")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--samples-per-epoch", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    for name, variant, latent_channels, train_snrs in EXPERIMENTS:
        checkpoint = PROJECT_DIR / "outputs" / "checkpoints" / f"{name}.pth"
        if args.force or not checkpoint.exists():
            command = [
                sys.executable,
                "train.py",
                "--name",
                name,
                "--variant",
                variant,
                "--latent-channels",
                str(latent_channels),
                "--epochs",
                str(args.epochs),
                "--samples-per-epoch",
                str(args.samples_per_epoch),
                "--device",
                args.device,
                "--train-snrs",
                *[str(value) for value in train_snrs],
            ]
            run(command)
        run(
            [
                sys.executable,
                "evaluate.py",
                "--checkpoint",
                str(checkpoint),
                "--repeats",
                str(args.repeats),
                "--device",
                args.device,
            ]
        )
    run([sys.executable, "plot_results.py"])


if __name__ == "__main__":
    main()
