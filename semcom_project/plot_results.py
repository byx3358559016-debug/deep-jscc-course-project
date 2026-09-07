"""Combine per-model summaries into comparison tables and publication-ready curves."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"


MODEL_ORDER = [
    "baseline_c16_random",
    "improved_c8_random",
    "improved_c16_random",
    "improved_c32_random",
    "improved_c16_fixed5",
]
LABELS = {
    "baseline_c16_random": "Baseline C16 / random SNR",
    "improved_c8_random": "Improved C8 / random SNR",
    "improved_c16_random": "Improved C16 / random SNR",
    "improved_c32_random": "Improved C32 / random SNR",
    "improved_c16_fixed5": "Improved C16 / fixed 5 dB",
}


def main() -> None:
    metrics_dir = OUTPUT_DIR / "metrics"
    curve_dir = OUTPUT_DIR / "curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(metrics_dir.glob("*_summary.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    rows = [row for row in rows if row["model"] in MODEL_ORDER]
    if not rows:
        raise RuntimeError("No experiment summary CSV files found")
    rows.sort(key=lambda row: (MODEL_ORDER.index(row["model"]), float(row["snr_db"])))

    comparison_path = metrics_dir / "model_comparison.csv"
    with comparison_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    specs = [
        ("psnr", "PSNR (dB)", "snr_psnr.png", False),
        ("ssim", "SSIM", "snr_ssim.png", False),
        ("mse", "MSE", "snr_mse.png", True),
    ]
    for metric, ylabel, filename, log_scale in specs:
        figure, axis = plt.subplots(figsize=(8.2, 5.1), constrained_layout=True)
        for model_name in MODEL_ORDER:
            model_rows = [row for row in rows if row["model"] == model_name]
            if not model_rows:
                continue
            x_values = [float(row["snr_db"]) for row in model_rows]
            means = [float(row[f"{metric}_mean"]) for row in model_rows]
            stds = [float(row[f"{metric}_std"]) for row in model_rows]
            axis.errorbar(x_values, means, yerr=stds, marker="o", linewidth=1.8, capsize=3, label=LABELS[model_name])
        axis.set_xlabel("Test SNR (dB)")
        axis.set_ylabel(ylabel)
        axis.set_title(f"Reconstruction {ylabel} under AWGN channel")
        axis.grid(True, linestyle="--", alpha=0.35)
        if log_scale:
            axis.set_yscale("log")
        axis.legend(fontsize=8)
        figure.savefig(curve_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.2, 5.1), constrained_layout=True)
    for model_name in MODEL_ORDER:
        log_path = OUTPUT_DIR / "logs" / f"{model_name}_training.csv"
        if not log_path.exists():
            continue
        with log_path.open("r", encoding="utf-8-sig", newline="") as stream:
            log_rows = list(csv.DictReader(stream))
        axis.plot(
            [int(row["epoch"]) for row in log_rows],
            [float(row["train_loss"]) for row in log_rows],
            linewidth=1.5,
            label=LABELS[model_name],
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training MSE loss")
    axis.set_title("Training loss of all experiment models")
    axis.set_yscale("log")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(fontsize=8)
    figure.savefig(curve_dir / "training_loss.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 3.6), constrained_layout=True)
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 3)
    axis.axis("off")
    boxes = [
        (0.25, 1.05, 1.3, 0.9, "RGB image\n3 x H x W", "#EAF2F8"),
        (2.0, 1.05, 1.55, 0.9, "CNN encoder\n2x downsample", "#D6EAF8"),
        (4.0, 1.05, 1.6, 0.9, "Residual +\npower normalize", "#D5F5E3"),
        (6.05, 1.05, 1.25, 0.9, "AWGN\nchannel", "#FDEBD0"),
        (7.75, 1.05, 1.55, 0.9, "CNN decoder\n2x upsample", "#D6EAF8"),
    ]
    for x, y, width, height, label, color in boxes:
        patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.04", facecolor=color, edgecolor="#34495E", linewidth=1.3)
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=10)
    for index in range(len(boxes) - 1):
        x1 = boxes[index][0] + boxes[index][2]
        x2 = boxes[index + 1][0]
        axis.annotate("", xy=(x2 - 0.05, 1.5), xytext=(x1 + 0.05, 1.5), arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#34495E"})
    axis.text(5.0, 2.55, "Lightweight Deep JSCC end-to-end image transmission", ha="center", fontsize=14, weight="bold")
    axis.text(4.8, 0.55, "Latent channels C = 8 / 16 / 32 determine bandwidth ratio k/n", ha="center", fontsize=10, color="#555555")
    figure.savefig(curve_dir / "system_architecture.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {comparison_path} and curves under {curve_dir}")


if __name__ == "__main__":
    main()
