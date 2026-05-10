"""
train.py — Train the discriminative U-Net speech denoiser
=========================================================

Usage:
    python scripts/train.py
    python scripts/train.py --clean_dir data/clean_speech --epochs 200 --batch_size 16

Loss:
    L1 (waveform) + Multi-Resolution STFT (spectral fidelity across 4 scales)
"""

import os
import sys
import time
import argparse

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import GradScaler, autocast

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from models.enhancer_unet import SpeechEnhancerUNet
from scripts.dataset import get_dataloader


# ── Multi-Resolution STFT Loss ──────────────────────────────────────────────

def stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int, device) -> torch.Tensor:
    """Compute STFT magnitude spectrogram."""
    window = torch.hann_window(win, device=device)
    return torch.stft(
        x, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, return_complex=True,
    ).abs()


def multi_stft_loss(clean: torch.Tensor, pred: torch.Tensor, device) -> torch.Tensor:
    """
    Multi-resolution STFT loss.

    For each (n_fft, hop, win) config, computes:
        L1(|STFT_pred|, |STFT_clean|) + MSE(log(1 + |STFT_pred|), log(1 + |STFT_clean|))

    This enforces both amplitude accuracy and log-scale perceptual similarity.
    """
    clean = clean.squeeze(1)
    pred = pred.squeeze(1)
    total = 0.0
    for n_fft, hop, win in Config.stft_configs:
        sc = stft_mag(clean, n_fft, hop, win, device)
        sp = stft_mag(pred, n_fft, hop, win, device)
        total += F.l1_loss(sp, sc) + F.mse_loss(sp.log1p(), sc.log1p())
    return total / len(Config.stft_configs)


# ── Training Loop ────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # Model
    model = SpeechEnhancerUNet(base_ch=args.base_ch).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Data
    loader = get_dataloader(
        clean_dir=args.clean_dir,
        batch_size=args.batch_size,
        noise_dir=args.noise_dir,
        sample_rate=Config.sample_rate,
        segment_length=Config.segment_length,
        snr_range=Config.snr_range,
    )
    print(f"Dataset: {len(loader.dataset)} clips | Batches/epoch: {len(loader)}")

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=Config.weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(loader),
        pct_start=0.1,
        anneal_strategy="cos",
    )
    scaler = GradScaler("cuda", enabled=(device.type == "cuda" and Config.use_amp))

    os.makedirs(os.path.dirname(args.checkpoint) or "checkpoints", exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        t0 = time.time()

        for clean, noisy in loader:
            clean = clean.to(device, non_blocking=True)
            noisy = noisy.to(device, non_blocking=True)

            with autocast("cuda", enabled=(device.type == "cuda" and Config.use_amp)):
                enhanced = model(noisy)
                loss_waveform = F.l1_loss(enhanced, clean)
                loss_spectral = multi_stft_loss(clean, enhanced, device)
                loss = loss_waveform + loss_spectral

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item()

        avg_loss = loss_sum / len(loader)
        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:3d}/{args.epochs}] | "
            f"Loss: {avg_loss:.4f} | LR: {lr_now:.2e} | Time: {elapsed:.1f}s"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.checkpoint)
            print(f"  → Best model saved (loss={best_loss:.4f})")

        if epoch % 25 == 0:
            path = f"checkpoints/unet_ep{epoch:03d}.pt"
            torch.save(model.state_dict(), path)
            print(f"  → Checkpoint: {path}")

    print(f"\nTraining complete. Best model: {args.checkpoint}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train U-Net speech denoiser")
    parser.add_argument("--clean_dir", default=Config.clean_dir, help="Clean speech WAV directory")
    parser.add_argument("--noise_dir", default=Config.noise_dir, help="Noise WAV directory (optional)")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--base_ch", type=int, default=Config.base_channels)
    parser.add_argument("--checkpoint", default=Config.checkpoint_path)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
