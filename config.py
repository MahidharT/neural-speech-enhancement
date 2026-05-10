"""
config.py — Central configuration for the speech enhancement pipeline.

All hyperparameters live here so experiments are reproducible.
Override via command-line args in train.py / enhance.py.
"""


class Config:
    # ── Audio ────────────────────────────────────────────────────────────────
    sample_rate: int = 16_000          # Working sample rate for Stage 1
    segment_length: int = 16_384       # ~1.024 s at 16 kHz — fits 8 GB VRAM

    # ── STFT ─────────────────────────────────────────────────────────────────
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 512

    # ── U-Net Model ──────────────────────────────────────────────────────────
    base_channels: int = 64            # Encoder widths: 64→128→256→512→1024
    bottleneck_blocks: int = 3         # ResBlocks in the bottleneck

    # ── Training ─────────────────────────────────────────────────────────────
    batch_size: int = 16
    epochs: int = 200
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    use_amp: bool = True               # Mixed-precision FP16

    # ── Noise augmentation ───────────────────────────────────────────────────
    snr_range: tuple = (0, 15)         # Uniform random SNR in dB
    noise_types: int = 4               # white, pink, real, babble

    # ── Multi-resolution STFT loss ───────────────────────────────────────────
    stft_configs: list = [
        (256,  32,  256),
        (512,  64,  512),
        (1024, 128, 1024),
        (2048, 256, 2048),
    ]

    # ── Paths ────────────────────────────────────────────────────────────────
    clean_dir: str = "data/clean_speech"
    noise_dir: str = "data/noise_only"
    checkpoint_path: str = "checkpoints/best_discriminative.pt"

    # ── Bandwidth Extension (AP-BWE Stage 2) ─────────────────────────────────
    bwe_checkpoint: str = "checkpoints/ap_bwe/g_01000000"
    bwe_target_sr: int = 48_000
