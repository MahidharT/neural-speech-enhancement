"""
utils.py — Helper functions for audio I/O, metrics, and visualization
=====================================================================
"""

import numpy as np
import torch
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Audio I/O ────────────────────────────────────────────────────────────────

def load_audio(path: str, sr: int = 16_000) -> np.ndarray:
    """Load a WAV file, convert to mono float32, resample if needed."""
    import torchaudio

    data, file_sr = sf.read(path)
    if data.ndim == 2:
        data = data.mean(axis=-1)
    data = data.astype(np.float32)

    if file_sr != sr:
        t = torch.from_numpy(data).float().unsqueeze(0)
        t = torchaudio.functional.resample(t, file_sr, sr).squeeze(0)
        data = t.numpy()

    return data


def save_audio(path: str, audio: np.ndarray, sr: int = 16_000):
    """Save a float32 numpy array as a 16-bit PCM WAV file."""
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(path, audio, sr, subtype="PCM_16")


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_snr(clean: np.ndarray, degraded: np.ndarray) -> float:
    """
    Signal-to-Noise Ratio (dB).

    SNR = 10 * log10( var(clean) / var(clean - degraded) )
    """
    err = clean[: len(degraded)] - degraded[: len(clean)]
    return float(10 * np.log10(np.var(clean).clip(1e-8) / np.var(err).clip(1e-8)))


def compute_si_snr(clean: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Scale-Invariant Signal-to-Noise Ratio (SI-SNR) in dB.

    Invariant to the scale of the estimated signal.
    Standard metric for source separation and speech enhancement.
    """
    clean = clean - np.mean(clean)
    enhanced = enhanced - np.mean(enhanced)

    # Optimal rescaling
    dot = np.sum(clean * enhanced)
    s_target = dot / (np.sum(clean ** 2) + 1e-8) * clean
    e_noise = enhanced - s_target

    si_snr = 10 * np.log10(
        np.sum(s_target ** 2).clip(1e-8) / np.sum(e_noise ** 2).clip(1e-8)
    )
    return float(si_snr)


# ── Visualization ────────────────────────────────────────────────────────────

def plot_comparison(
    noisy: np.ndarray,
    enhanced: np.ndarray,
    sr: int = 16_000,
    save_path: str = None,
):
    """
    Side-by-side waveform + spectrogram comparison.

    Args:
        noisy:     Noisy waveform (1-D numpy array)
        enhanced:  Enhanced waveform (1-D numpy array)
        sr:        Sample rate
        save_path: If given, saves the figure to this path
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor="#0d0d1a")

    for ax in axes.flat:
        ax.set_facecolor("#13132b")
        ax.tick_params(colors="#8888aa", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e1e3a")

    t_noisy = np.arange(len(noisy)) / sr
    t_enh = np.arange(len(enhanced)) / sr

    # Noisy waveform
    axes[0, 0].plot(t_noisy, noisy, color="#ff6b6b", linewidth=0.5, alpha=0.9)
    axes[0, 0].set_title("Noisy Waveform", color="#ff6b6b", fontsize=10, fontweight="bold")
    axes[0, 0].set_ylim(-1.1, 1.1)

    # Enhanced waveform
    axes[0, 1].plot(t_enh, enhanced, color="#00d4aa", linewidth=0.5, alpha=0.9)
    axes[0, 1].set_title("Enhanced Waveform", color="#00d4aa", fontsize=10, fontweight="bold")
    axes[0, 1].set_ylim(-1.1, 1.1)

    # Noisy spectrogram
    axes[1, 0].specgram(noisy, Fs=sr, NFFT=512, noverlap=384, cmap="magma", scale="dB")
    axes[1, 0].set_title("Noisy Spectrogram", color="#ff6b6b", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Hz", color="#8888aa", fontsize=8)

    # Enhanced spectrogram
    axes[1, 1].specgram(enhanced, Fs=sr, NFFT=512, noverlap=384, cmap="magma", scale="dB")
    axes[1, 1].set_title("Enhanced Spectrogram", color="#00d4aa", fontsize=10, fontweight="bold")
    axes[1, 1].set_ylabel("Hz", color="#8888aa", fontsize=8)

    fig.suptitle(
        "Speech Enhancement — Before vs After",
        color="#e8e8ff", fontsize=14, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Figure saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)
