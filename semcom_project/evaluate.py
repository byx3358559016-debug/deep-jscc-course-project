"""Evaluate a trained Deep JSCC checkpoint over SNR and save reproducible artifacts."""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from data import KodakImageDataset, load_split
from model import build_model
from utils import DEFAULT_DATA_DIR, DEFAULT_SPLIT_FILE, OUTPUT_DIR, choose_device, compute_metrics, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--snrs", type=float, nargs="+", default=[-10, -5, 0, 5, 10])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def _pil_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().clamp(0, 1).mul(255).byte().numpy().transpose(1, 2, 0)
    return Image.fromarray(array, mode="RGB")


def save_reconstruction_grid(dataset, snrs, reconstructions, output_path: Path, model_name: str) -> None:
    cell_width, image_height, label_height = 270, 190, 48
    top_height = 42
    canvas = Image.new("RGB", (cell_width * (len(snrs) + 1), top_height + 2 * (image_height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 19)
        label_font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 14)
    except OSError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    draw.text((canvas.width // 2, 20), f"{model_name}: original and reconstructed test images", fill="black", font=title_font, anchor="mm")
    for row_index in range(2):
        original, image_name = dataset[row_index]
        entries = [(original, f"{image_name}\nOriginal")]
        for snr_db in snrs:
            reconstruction, metrics = reconstructions[(image_name, snr_db)]
            entries.append((reconstruction, f"{snr_db:g} dB\n{metrics['psnr']:.2f} dB / {metrics['ssim']:.3f}"))
        row_top = top_height + row_index * (image_height + label_height)
        for column_index, (tensor, label) in enumerate(entries):
            image = ImageOps.contain(_pil_image(tensor), (cell_width - 12, image_height - 8))
            left = column_index * cell_width + (cell_width - image.width) // 2
            top = row_top + (image_height - image.height) // 2
            canvas.paste(image, (left, top))
            draw.multiline_text(
                (column_index * cell_width + cell_width // 2, row_top + image_height + 4),
                label,
                fill="black",
                font=label_font,
                anchor="ma",
                align="center",
                spacing=2,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, dpi=(180, 180))


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    name = config["name"]
    set_seed(int(config.get("seed", 42)))
    model = build_model(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    split = load_split(args.split_file)
    dataset = KodakImageDataset(args.data_dir, split["test"])
    raw_rows = []
    first_reconstructions: dict[tuple[str, float], tuple[torch.Tensor, dict]] = {}

    with torch.no_grad():
        for snr_index, snr_db in enumerate(args.snrs):
            for image_index, (image, image_name) in enumerate(dataset):
                input_tensor = image.unsqueeze(0).to(device)
                for repeat in range(args.repeats):
                    seed = int(config.get("seed", 42)) + snr_index * 10_000 + image_index * 100 + repeat
                    torch.manual_seed(seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(seed)
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    reconstruction = model(input_tensor, snr_db, config.get("channel", "awgn"))
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    metrics = compute_metrics(image, reconstruction.squeeze(0))
                    raw_rows.append(
                        {
                            "model": name,
                            "image": image_name,
                            "snr_db": snr_db,
                            "repeat": repeat,
                            **metrics,
                            "inference_ms": elapsed_ms,
                        }
                    )
                    if repeat == 0 and image_index < 2:
                        first_reconstructions[(image_name, snr_db)] = (reconstruction.squeeze(0).cpu(), metrics)

    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in raw_rows:
        grouped[float(row["snr_db"])].append(row)
    summary_rows = []
    for snr_db in args.snrs:
        rows = grouped[float(snr_db)]
        summary = {
            "model": name,
            "variant": config["variant"],
            "latent_channels": config["latent_channels"],
            "residual_blocks": config.get("residual_blocks", 0),
            "power_normalization": config.get("power_normalization", False),
            "train_snrs": "/".join(str(value) for value in config["train_snrs"]),
            "train_channel": config.get("channel", "awgn"),
            "snr_db": snr_db,
            "samples": len(rows),
            "parameters": config["parameters"],
            "bandwidth_ratio": config["bandwidth_ratio"],
            "training_seconds": config["training_seconds"],
        }
        for metric in ("mse", "psnr", "ssim", "inference_ms"):
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(summary)

    metrics_dir = OUTPUT_DIR / "metrics"
    reconstruction_dir = OUTPUT_DIR / "reconstructions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    raw_path = metrics_dir / f"{name}_raw.csv"
    summary_path = metrics_dir / f"{name}_summary.csv"
    with raw_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)
    with summary_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    save_reconstruction_grid(
        dataset,
        args.snrs,
        first_reconstructions,
        reconstruction_dir / f"{name}_comparison.png",
        name,
    )

    print(f"Saved {raw_path}")
    print(f"Saved {summary_path}")
    for row in summary_rows:
        print(
            f"SNR={row['snr_db']:>5g} dB  MSE={row['mse_mean']:.5f}  "
            f"PSNR={row['psnr_mean']:.2f} dB  SSIM={row['ssim_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
