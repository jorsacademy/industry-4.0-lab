# Data

Raw files are intentionally not versioned. Run `python scripts/download_data.py` to hydrate the public Kaggle release into `data/raw/` and generate a SHA-256 manifest.

The benchmark consumes the published frequency-feature table and keeps the original time-series/audio files available for future signal-processing extensions.
