"""Benchmark checkpoint size and CPU/GPU inference latency on one Kodak test image."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch

from data import KodakImageDataset, load_split
from model import build_model
from utils import DEFAULT_DATA_DIR, DEFAULT_SPLIT_FILE, OUTPUT_DIR


@torch.no_grad()
def latency_ms(model, image, device: torch.device, warmup: int = 3, repeats: int = 10) -> float:
    model = model.to(device).eval()
    image = image.unsqueeze(0).to(device)
    for _ in range(warmup):
        model(image, 5.0, "awgn")
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        model(image, 5.0, "awgn")
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / repeats


def main() -> None:
    split = load_split(DEFAULT_SPLIT_FILE)
    image, image_name = KodakImageDataset(DEFAULT_DATA_DIR, split["test"])[0]
    rows = []
    for checkpoint_path in sorted((OUTPUT_DIR / "checkpoints").glob("*.pth")):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = payload["config"]
        model = build_model(config)
        model.load_state_dict(payload["state_dict"])
        cpu_ms = latency_ms(model, image, torch.device("cpu"))
        gpu_ms = float("nan")
        if torch.cuda.is_available():
            model = build_model(config)
            model.load_state_dict(payload["state_dict"])
            gpu_ms = latency_ms(model, image, torch.device("cuda"))
        rows.append(
            {
                "model": config["name"],
                "test_image": image_name,
                "image_shape": "x".join(str(value) for value in image.shape),
                "parameters": config["parameters"],
                "bandwidth_ratio": config["bandwidth_ratio"],
                "checkpoint_kib": checkpoint_path.stat().st_size / 1024.0,
                "training_seconds": config["training_seconds"],
                "best_epoch": config["best_epoch"],
                "cpu_inference_ms": cpu_ms,
                "gpu_inference_ms": gpu_ms,
                "cpu_threads": torch.get_num_threads(),
            }
        )
        print(f"{config['name']}: CPU={cpu_ms:.2f}ms GPU={gpu_ms:.2f}ms")
    output_path = OUTPUT_DIR / "metrics" / "complexity.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
