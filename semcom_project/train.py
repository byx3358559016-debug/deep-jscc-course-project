"""Train a leakage-free lightweight Deep JSCC model."""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import KodakImageDataset, KodakPatchDataset, load_split
from model import SemanticCommSystem
from utils import DEFAULT_DATA_DIR, DEFAULT_SPLIT_FILE, OUTPUT_DIR, bandwidth_ratio, choose_device, count_parameters, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--variant", choices=["baseline", "improved"], default="baseline")
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--train-snrs", type=float, nargs="+", default=[-10, -5, 0, 5, 10])
    parser.add_argument("--channel", choices=["awgn", "rayleigh"], default="awgn")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--samples-per-epoch", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


@torch.no_grad()
def validation_psnr(model, dataset, device, snr_db: float, channel: str, seed: int) -> tuple[float, float]:
    model.eval()
    mses = []
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        for image, _ in dataset:
            image = image.unsqueeze(0).to(device)
            reconstruction = model(image, snr_db, channel)
            mses.append(float(nn.functional.mse_loss(reconstruction, image).item()))
    mse = sum(mses) / len(mses)
    psnr = 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()
    return mse, psnr


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    split = load_split(args.split_file)
    train_dataset = KodakPatchDataset(
        args.data_dir,
        split["train"],
        patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch,
        seed=args.seed,
    )
    val_dataset = KodakImageDataset(args.data_dir, split["val"])
    loader_generator = torch.Generator().manual_seed(args.seed)
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
        pin_memory=device.type == "cuda",
    )

    residual_blocks = 1 if args.variant == "improved" else 0
    power_normalization = args.variant == "improved"
    model = SemanticCommSystem(args.latent_channels, residual_blocks, power_normalization).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    checkpoints = OUTPUT_DIR / "checkpoints"
    logs = OUTPUT_DIR / "logs"
    checkpoints.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{args.name}_training.csv"

    best_psnr = float("-inf")
    best_state = None
    best_epoch = 0
    stale_checks = 0
    rows = []
    started = time.perf_counter()
    print(f"Training {args.name} on {device}; parameters={count_parameters(model):,}")

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device, non_blocking=True)
            snr_db = random.choice(args.train_snrs)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch, snr_db, args.channel)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch.shape[0]
        train_loss = total_loss / len(train_dataset)

        val_mse = float("nan")
        val_psnr = float("nan")
        if epoch == 1 or epoch % args.val_every == 0 or epoch == args.epochs:
            validation_snr = 5.0 if len(args.train_snrs) > 1 else args.train_snrs[0]
            val_mse, val_psnr = validation_psnr(
                model, val_dataset, device, validation_snr, args.channel, args.seed + 2026
            )
            if val_psnr > best_psnr + 1e-4:
                best_psnr = val_psnr
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale_checks = 0
            else:
                stale_checks += 1
            print(
                f"epoch={epoch:03d} loss={train_loss:.6f} "
                f"val_psnr={val_psnr:.3f}dB best={best_psnr:.3f}dB"
            )
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_mse": val_mse, "val_psnr": val_psnr})
        if stale_checks >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    training_seconds = time.perf_counter() - started
    if best_state is None:
        best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        best_epoch = len(rows)
        best_psnr = float("nan")

    config = {
        "name": args.name,
        "variant": args.variant,
        "latent_channels": args.latent_channels,
        "residual_blocks": residual_blocks,
        "power_normalization": power_normalization,
        "train_snrs": args.train_snrs,
        "channel": args.channel,
        "patch_size": args.patch_size,
        "samples_per_epoch": args.samples_per_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "split_file": str(args.split_file.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "parameters": count_parameters(model),
        "bandwidth_ratio": bandwidth_ratio(args.latent_channels),
        "best_epoch": best_epoch,
        "best_val_psnr": best_psnr,
        "training_seconds": training_seconds,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    checkpoint_path = checkpoints / f"{args.name}.pth"
    torch.save({"state_dict": best_state, "config": config}, checkpoint_path)
    with log_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "val_mse", "val_psnr"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Training time: {training_seconds:.2f}s; best epoch: {best_epoch}")


if __name__ == "__main__":
    main()
