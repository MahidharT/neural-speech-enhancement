"""
evaluate.py — Compute objective speech quality metrics
======================================================

Metrics:
    - PESQ  (Perceptual Evaluation of Speech Quality, ITU-T P.862)
    - STOI  (Short-Time Objective Intelligibility)
    - SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
    - SNR   (Signal-to-Noise Ratio)

Usage:
    python scripts/evaluate.py --clean_dir data/clean_speech --enhanced_dir results/audio_samples
    python scripts/evaluate.py --clean file.wav --enhanced enhanced.wav
"""

import os
import sys
import argparse
import glob

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from scripts.utils import load_audio, compute_snr, compute_si_snr


def compute_pesq_score(clean: np.ndarray, enhanced: np.ndarray, sr: int) -> float:
    """Compute PESQ score. Requires the 'pesq' package."""
    try:
        from pesq import pesq
        # PESQ supports 8000 or 16000 Hz
        mode = "wb" if sr == 16000 else "nb"
        return pesq(sr, clean, enhanced, mode)
    except ImportError:
        print("  [WARN] 'pesq' not installed. Run: pip install pesq")
        return float("nan")
    except Exception as e:
        print(f"  [WARN] PESQ failed: {e}")
        return float("nan")


def compute_stoi_score(clean: np.ndarray, enhanced: np.ndarray, sr: int) -> float:
    """Compute STOI score. Requires the 'pystoi' package."""
    try:
        from pystoi import stoi
        return stoi(clean, enhanced, sr, extended=False)
    except ImportError:
        print("  [WARN] 'pystoi' not installed. Run: pip install pystoi")
        return float("nan")
    except Exception as e:
        print(f"  [WARN] STOI failed: {e}")
        return float("nan")


def evaluate_pair(clean_path: str, enhanced_path: str, sr: int = Config.sample_rate) -> dict:
    """Evaluate a single clean/enhanced audio pair."""
    clean = load_audio(clean_path, sr=sr)
    enhanced = load_audio(enhanced_path, sr=sr)

    # Align lengths
    min_len = min(len(clean), len(enhanced))
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]

    metrics = {
        "file": os.path.basename(enhanced_path),
        "SNR_dB": compute_snr(clean, enhanced),
        "SI-SNR_dB": compute_si_snr(clean, enhanced),
        "PESQ": compute_pesq_score(clean, enhanced, sr),
        "STOI": compute_stoi_score(clean, enhanced, sr),
    }
    return metrics


def evaluate_directory(clean_dir: str, enhanced_dir: str, sr: int = Config.sample_rate):
    """Evaluate all matching WAV pairs in two directories."""
    enhanced_files = sorted(glob.glob(os.path.join(enhanced_dir, "*.wav")))
    if not enhanced_files:
        print(f"No WAV files found in {enhanced_dir}")
        return []

    all_metrics = []
    for enh_path in enhanced_files:
        basename = os.path.basename(enh_path)
        clean_path = os.path.join(clean_dir, basename)

        if not os.path.exists(clean_path):
            print(f"  Skipping {basename} — no matching clean file")
            continue

        print(f"  Evaluating: {basename}")
        metrics = evaluate_pair(clean_path, enh_path, sr)
        all_metrics.append(metrics)

        for k, v in metrics.items():
            if k != "file":
                print(f"    {k}: {v:.4f}")

    # Summary
    if all_metrics:
        print("\n" + "=" * 60)
        print("AVERAGE METRICS")
        print("=" * 60)
        for key in ["SNR_dB", "SI-SNR_dB", "PESQ", "STOI"]:
            values = [m[key] for m in all_metrics if not np.isnan(m[key])]
            if values:
                print(f"  {key:>10s}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        print("=" * 60)

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate speech enhancement quality")
    parser.add_argument("--clean", help="Clean WAV file or directory")
    parser.add_argument("--enhanced", help="Enhanced WAV file or directory")
    parser.add_argument("--clean_dir", help="Directory of clean WAVs")
    parser.add_argument("--enhanced_dir", help="Directory of enhanced WAVs")
    args = parser.parse_args()

    if args.clean and args.enhanced:
        # Single-file evaluation
        metrics = evaluate_pair(args.clean, args.enhanced)
        print("\nResults:")
        for k, v in metrics.items():
            if k != "file":
                print(f"  {k}: {v:.4f}")
    elif args.clean_dir and args.enhanced_dir:
        # Directory evaluation
        evaluate_directory(args.clean_dir, args.enhanced_dir)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/evaluate.py --clean clean.wav --enhanced enhanced.wav")
        print("  python scripts/evaluate.py --clean_dir data/clean --enhanced_dir results/audio_samples")


if __name__ == "__main__":
    main()
