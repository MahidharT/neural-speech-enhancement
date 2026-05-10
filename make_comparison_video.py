"""
make_comparison_video.py — Generate a before/after demo video
=============================================================

Creates a 20-second video with the same audio segment:
  0–10 s : Noisy signal (red)
  10–20 s: Enhanced signal (green)

Includes real-time scrolling waveform + spectrogram with audio.

Usage:
    python scripts/make_comparison_video.py
"""

import os
import sys
import glob
import random

import numpy as np
import torch
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio
import subprocess
import imageio_ffmpeg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from models.enhancer_unet import SpeechEnhancerUNet
from scripts.dataset import pink_noise

# ── Settings ─────────────────────────────────────────────────────────────────
SR = Config.sample_rate
AUDIO_LEN_SEC = 10
DURATION_SEC = 20
CHUNK_LEN = Config.segment_length
FPS = 24
CHECKPOINT = Config.checkpoint_path
OUT_VIDEO = "results/figures/comparison_video.mp4"
CLEAN_DIR = Config.clean_dir

# ── 1. Build audio ──────────────────────────────────────────────────────────
print("Step 1/4 — Loading clean audio ...")
wav_files = sorted(glob.glob(os.path.join(CLEAN_DIR, "*.wav")))
assert wav_files, f"No WAVs in {CLEAN_DIR}/"

target_samples = SR * AUDIO_LEN_SEC
clean_chunks = []
for path in random.sample(wav_files, min(len(wav_files), 30)):
    data, file_sr = sf.read(path)
    if data.ndim == 2:
        data = data.mean(axis=-1)
    if file_sr != SR:
        n = int(len(data) * SR / file_sr)
        data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data)
    clean_chunks.append(data)
    if sum(len(c) for c in clean_chunks) >= target_samples:
        break

clean_all = np.concatenate(clean_chunks)[:target_samples].astype(np.float32)
clean_all /= np.abs(clean_all).max() + 1e-8

print("Step 1/4 — Adding noise at 5 dB SNR ...")
sig_pwr = np.var(clean_all).clip(1e-8)
nse_pwr = sig_pwr / (10 ** (5.0 / 10))
noise = np.random.randn(len(clean_all)).astype(np.float32)
noise *= np.sqrt(nse_pwr / np.var(noise).clip(1e-8))
noisy_all = np.clip(clean_all + noise, -1, 1)

# ── 2. Enhance ──────────────────────────────────────────────────────────────
print("Step 2/4 — Running U-Net denoiser ...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpeechEnhancerUNet(base_ch=Config.base_channels).to(device)
if os.path.exists(CHECKPOINT):
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    print(f"  Loaded: {CHECKPOINT}")
model.eval()

enhanced_chunks = []
for i in range(int(np.ceil(len(noisy_all) / CHUNK_LEN))):
    s, e = i * CHUNK_LEN, min((i + 1) * CHUNK_LEN, len(noisy_all))
    chunk = noisy_all[s:e]
    pad = CHUNK_LEN - len(chunk)
    if pad > 0:
        chunk = np.concatenate([chunk, np.zeros(pad, np.float32)])
    inp = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp).cpu().squeeze().numpy()
    enhanced_chunks.append(out[: CHUNK_LEN - pad] if pad else out)

enhanced_all = np.concatenate(enhanced_chunks)[:target_samples].astype(np.float32)
enhanced_all /= np.abs(enhanced_all).max() + 1e-8

snr_n = 10 * np.log10(np.var(clean_all).clip(1e-8) / np.var(clean_all - noisy_all).clip(1e-8))
snr_e = 10 * np.log10(np.var(clean_all).clip(1e-8) / np.var(clean_all - enhanced_all).clip(1e-8))
print(f"  SNR noisy:    {snr_n:.2f} dB")
print(f"  SNR enhanced: {snr_e:.2f} dB")

# ── 3. Render frames ────────────────────────────────────────────────────────
print("Step 3/4 — Rendering frames ...")
TOTAL = FPS * DURATION_SEC
WIN = int(SR * 0.5)
BG, PBG = "#0d0d1a", "#13132b"
CN, CE, CG, W, G = "#ff6b6b", "#00d4aa", "#1e1e3a", "#e8e8ff", "#8888aa"


def frame(fi):
    t = fi / FPS
    is_noisy = t < AUDIO_LEN_SEC
    lt = t if is_noisy else t - AUDIO_LEN_SEC
    sp = int(lt * SR)
    sig = noisy_all if is_noisy else enhanced_all
    col = CN if is_noisy else CE
    lbl = "NOISY SIGNAL" if is_noisy else "ENHANCED SIGNAL"
    ws = max(0, sp - WIN // 2)
    we = min(ws + WIN, len(sig))
    ws = max(0, we - WIN)

    fig = plt.figure(figsize=(10, 8), facecolor=BG)
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.4, left=0.1, right=0.9, top=0.85, bottom=0.1)
    fig.text(0.5, 0.94, "Speech Enhancement", ha="center", color=W, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.90, f"NOW PLAYING: {lbl}", ha="center", color=col, fontsize=12, fontweight="bold")

    bar = fig.add_axes([0.1, 0.88, 0.8, 0.015])
    bar.set_facecolor(CG)
    bar.barh(0, t / DURATION_SEC, color=col, height=1)
    bar.set_xlim(0, 1); bar.set_ylim(-0.5, 0.5); bar.axis("off")

    ax = fig.add_subplot(gs[0])
    ta = np.linspace(ws / SR, we / SR, we - ws)
    ax.plot(ta, sig[ws:we], color=col, lw=0.9, alpha=0.95)
    ax.axvline(sp / SR, color="white", lw=1.2, alpha=0.8)
    ax.set_facecolor(PBG); ax.set_ylim(-1.1, 1.1); ax.set_title("Waveform", color=col, fontsize=9, fontweight="bold")
    ax.tick_params(colors=G, labelsize=6)

    ax2 = fig.add_subplot(gs[1])
    ax2.specgram(sig, Fs=SR, NFFT=256, noverlap=200, cmap="magma", scale="dB", vmin=-80, vmax=0)
    ax2.axvline(sp / SR, color="white", lw=1.2, alpha=0.8)
    ax2.set_facecolor(PBG); ax2.set_title("Spectrogram", color=col, fontsize=9, fontweight="bold")
    ax2.tick_params(colors=G, labelsize=6)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[:, :, :3].copy()
    plt.close(fig)
    return img


frames = []
for fi in range(TOTAL):
    if fi % FPS == 0:
        print(f"  Frame {fi}/{TOTAL} ({fi // FPS}s)")
    frames.append(frame(fi))

# ── 4. Encode ────────────────────────────────────────────────────────────────
print("Step 4/4 — Encoding MP4 ...")
os.makedirs(os.path.dirname(OUT_VIDEO), exist_ok=True)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
TMP = "tmp_vid.mp4"
w = imageio.get_writer(TMP, format="ffmpeg", fps=FPS, quality=8, macro_block_size=1)
for f in frames:
    w.append_data(f)
w.close()

audio = np.concatenate([noisy_all, enhanced_all]).astype(np.float32)
sf.write("tmp_audio.wav", audio, SR)
subprocess.run([FFMPEG, "-y", "-i", TMP, "-i", "tmp_audio.wav", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", OUT_VIDEO], capture_output=True)
for f in [TMP, "tmp_audio.wav"]:
    try:
        os.remove(f)
    except Exception:
        pass

print(f"\nDone! → {OUT_VIDEO}")
print(f"SNR noisy={snr_n:.2f} dB | SNR enhanced={snr_e:.2f} dB")
