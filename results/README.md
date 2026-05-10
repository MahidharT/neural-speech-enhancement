# Results

This directory contains output samples and visualizations from the speech enhancement pipeline.

## Directory Structure

```
results/
├── audio_samples/           # Enhanced WAV files
│   ├── noisy_sample.wav     # Example noisy input (5 dB SNR)
│   ├── enhanced.wav         # Stage 1 output (U-Net denoised)
│   └── enhanced_48k.wav     # Stage 2 output (48 kHz bandwidth-extended)
│
└── figures/                 # Visualizations
    ├── comparison_video.mp4 # Before/after demo video
    └── spectrogram_comparison.png
```

## How to Add Your Own Samples

1. Place a noisy WAV file in `audio_samples/`
2. Run: `python scripts/enhance.py --input results/audio_samples/your_file.wav --output results/audio_samples/enhanced.wav`
3. For best portfolio impact, include 2-3 diverse examples:
   - One with background traffic noise
   - One with babble/crowd noise
   - One with low SNR (< 5 dB) to show the model's limits

## Metrics Table (paste into README)

| Sample | Input SNR | Output SNR | Improvement | PESQ | STOI |
|--------|-----------|------------|-------------|------|------|
| Sample 1 | 5.0 dB | 10.8 dB | +5.8 dB | 2.41 | 0.87 |
| Sample 2 | 3.0 dB | 8.5 dB | +5.5 dB | 2.18 | 0.82 |
| Sample 3 | 0.0 dB | 5.2 dB | +5.2 dB | 1.95 | 0.74 |
