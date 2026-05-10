"""
dataset.py — Speech enhancement dataset with on-the-fly noise augmentation
===========================================================================

Loads clean speech WAVs and mixes them with diverse noise at random SNRs.

Noise types:
    0 — White Gaussian noise
    1 — Pink (1/f) noise
    2 — Real environmental noise recordings (if available)
    3 — Babble noise (random clean speech overlaid)
"""

import os
import glob
import torch
import torchaudio
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader


def pink_noise(size: tuple) -> torch.Tensor:
    """Generate approximate pink (1/f) noise via spectral shaping."""
    white = torch.randn(size)
    f = torch.fft.rfftfreq(size[-1]).clamp(min=1e-6)
    power = (1.0 / f) ** 0.5
    pink = torch.fft.irfft(torch.fft.rfft(white) * power, n=size[-1])
    return pink / pink.std().clamp(min=1e-8)


class SpeechEnhancementDataset(Dataset):
    """
    On-the-fly noisy speech generation for training.

    For each clean clip, a random noise type and SNR are selected,
    the noise is scaled to match the target SNR, and the mixture
    is returned alongside the clean target.

    Args:
        clean_dir:      Directory containing clean *.wav files
        noise_dir:      Optional directory with real noise *.wav files
        sample_rate:    Working sample rate (default 16000)
        segment_length: Segment length in samples (default 16384 ≈ 1.024 s)
        snr_range:      (min_snr_db, max_snr_db) for uniform sampling
    """

    def __init__(
        self,
        clean_dir: str,
        noise_dir: str = None,
        sample_rate: int = 16_000,
        segment_length: int = 16_384,
        snr_range: tuple = (0, 15),
    ):
        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
        assert len(self.clean_files) > 0, f"No WAV files found in {clean_dir}"

        self.noise_files = []
        if noise_dir and os.path.isdir(noise_dir):
            self.noise_files = sorted(glob.glob(os.path.join(noise_dir, "*.wav")))

        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.snr_range = snr_range

    def __len__(self) -> int:
        return len(self.clean_files)

    def _load_and_prepare(self, path: str) -> torch.Tensor:
        """Load a WAV, convert to mono, resample, and crop/pad to segment_length."""
        data, sr = sf.read(path)
        wav = torch.from_numpy(data).float()

        if wav.dim() == 2:
            wav = wav.mean(dim=-1)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(
                wav.unsqueeze(0), sr, self.sample_rate
            ).squeeze(0)

        if wav.shape[-1] >= self.segment_length:
            start = torch.randint(0, wav.shape[-1] - self.segment_length + 1, (1,)).item()
            wav = wav[start : start + self.segment_length]
        else:
            wav = torch.nn.functional.pad(wav, (0, self.segment_length - wav.shape[-1]))

        wav = wav / wav.abs().max().clamp(min=1e-8)
        return wav

    def __getitem__(self, idx: int):
        clean_wav = self._load_and_prepare(self.clean_files[idx])

        # Random SNR
        snr_db = np.random.uniform(*self.snr_range)
        snr_linear = 10 ** (snr_db / 10)
        signal_power = clean_wav.var().clamp(min=1e-8)
        noise_power = signal_power / snr_linear

        # Random noise type
        noise_type = torch.randint(0, 4, (1,)).item()

        if noise_type == 0:
            noise = torch.randn_like(clean_wav)

        elif noise_type == 1:
            noise = pink_noise(clean_wav.shape)

        elif noise_type == 2 and self.noise_files:
            n_idx = torch.randint(0, len(self.noise_files), (1,)).item()
            noise = self._load_and_prepare(self.noise_files[n_idx])

        else:  # Babble
            b_idx = torch.randint(0, len(self.clean_files), (1,)).item()
            noise = self._load_and_prepare(self.clean_files[b_idx])

        # Scale noise to target SNR
        noise = noise * torch.sqrt(noise_power / noise.var().clamp(min=1e-8))
        noisy_wav = (clean_wav + noise).clamp(-1.0, 1.0)

        # Return [1, T] tensors
        return clean_wav.unsqueeze(0), noisy_wav.unsqueeze(0)


def get_dataloader(
    clean_dir: str,
    batch_size: int = 16,
    noise_dir: str = None,
    sample_rate: int = 16_000,
    segment_length: int = 16_384,
    snr_range: tuple = (0, 15),
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader with the speech enhancement dataset."""
    dataset = SpeechEnhancementDataset(
        clean_dir=clean_dir,
        noise_dir=noise_dir,
        sample_rate=sample_rate,
        segment_length=segment_length,
        snr_range=snr_range,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )
