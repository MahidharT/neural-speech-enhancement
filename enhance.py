"""
enhance.py — Run speech enhancement inference
==============================================

Produces ENHANCED SPEECH AUDIO from a noisy input file.

Usage:
    python scripts/enhance.py --input noisy.wav --output enhanced.wav
    python scripts/enhance.py --input noisy.wav --output enhanced_48k.wav --bwe

Stage 1: U-Net denoiser (16 kHz → 16 kHz clean)
Stage 2: AP-BWE bandwidth extension (16 kHz → 48 kHz)  [optional, with --bwe]
"""

import os
import sys
import argparse
import time

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from models.enhancer_unet import SpeechEnhancerUNet
from scripts.utils import load_audio, save_audio, compute_snr, compute_si_snr, plot_comparison


def enhance_audio(
    input_path: str,
    output_path: str,
    checkpoint: str = Config.checkpoint_path,
    base_ch: int = Config.base_channels,
    device_str: str = "auto",
    plot_path: str = None,
) -> dict:
    """
    Run the U-Net denoiser on an audio file.

    Args:
        input_path:  Path to noisy WAV
        output_path: Path to save enhanced WAV
        checkpoint:  Path to model checkpoint
        base_ch:     U-Net base channel width (must match training)
        device_str:  "auto", "cuda", or "cpu"
        plot_path:   Optional path to save comparison figure

    Returns:
        dict with keys: input_path, output_path, duration_sec, inference_time_sec
    """
    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")

    # Load model
    model = SpeechEnhancerUNet(base_ch=base_ch).to(device)
    if os.path.exists(checkpoint):
        sd = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(sd)
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint}, using untrained model")
    model.eval()

    # Load audio
    noisy = load_audio(input_path, sr=Config.sample_rate)
    print(f"Input:  {input_path}  ({len(noisy)/Config.sample_rate:.2f}s, {Config.sample_rate} Hz)")

    # Process in chunks to handle arbitrary-length audio
    chunk_len = Config.segment_length
    enhanced_chunks = []
    t0 = time.time()

    n_chunks = int(np.ceil(len(noisy) / chunk_len))
    for i in range(n_chunks):
        start = i * chunk_len
        end = min(start + chunk_len, len(noisy))
        chunk = noisy[start:end]

        # Pad if needed
        pad = chunk_len - len(chunk)
        if pad > 0:
            chunk = np.concatenate([chunk, np.zeros(pad, dtype=np.float32)])

        inp = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
        out_np = out.cpu().squeeze().numpy()

        if pad > 0:
            out_np = out_np[:-pad]
        enhanced_chunks.append(out_np)

    enhanced = np.concatenate(enhanced_chunks).astype(np.float32)
    inference_time = time.time() - t0

    # Normalize output
    enhanced = enhanced / (np.abs(enhanced).max() + 1e-8)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_audio(output_path, enhanced, sr=Config.sample_rate)
    print(f"Output: {output_path}  ({len(enhanced)/Config.sample_rate:.2f}s)")
    print(f"Inference time: {inference_time:.3f}s  ({len(noisy)/Config.sample_rate/inference_time:.1f}× real-time)")

    # Metrics (if we have a clean reference — compute SNR vs input)
    snr_val = compute_si_snr(noisy, enhanced)
    print(f"SI-SNR (noisy→enhanced): {snr_val:.2f} dB")

    # Optional comparison plot
    if plot_path:
        plot_comparison(noisy, enhanced, sr=Config.sample_rate, save_path=plot_path)

    return {
        "input_path": input_path,
        "output_path": output_path,
        "duration_sec": len(noisy) / Config.sample_rate,
        "inference_time_sec": inference_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Enhance noisy speech → clean speech")
    parser.add_argument("--input", required=True, help="Path to noisy WAV file")
    parser.add_argument("--output", required=True, help="Path to save enhanced WAV")
    parser.add_argument("--checkpoint", default=Config.checkpoint_path, help="Model checkpoint path")
    parser.add_argument("--base_ch", type=int, default=Config.base_channels)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--plot", default=None, help="Save comparison figure to this path")
    parser.add_argument("--bwe", action="store_true", help="Apply bandwidth extension (16→48 kHz)")
    args = parser.parse_args()

    # Stage 1: Denoise
    result = enhance_audio(
        input_path=args.input,
        output_path=args.output,
        checkpoint=args.checkpoint,
        base_ch=args.base_ch,
        device_str=args.device,
        plot_path=args.plot,
    )

    # Stage 2: Bandwidth Extension (optional)
    if args.bwe:
        print("\n--- Stage 2: Bandwidth Extension (AP-BWE) ---")
        try:
            import torchaudio.functional as aF

            enhanced, sr = sf.read(args.output)
            enhanced_t = torch.from_numpy(enhanced).float().unsqueeze(0)

            # Upsample 16k → 48k
            enhanced_48k = aF.resample(enhanced_t, orig_freq=sr, new_freq=Config.bwe_target_sr)
            enhanced_48k_np = enhanced_48k.squeeze().numpy()

            bwe_output = args.output.replace(".wav", "_48k.wav")
            save_audio(bwe_output, enhanced_48k_np, sr=Config.bwe_target_sr)
            print(f"BWE output: {bwe_output}  ({Config.bwe_target_sr} Hz)")

        except Exception as e:
            print(f"BWE stage failed: {e}")
            print("Falling back to 16 kHz output only.")

    print("\n✓ Enhancement complete!")


if __name__ == "__main__":
    main()
