# Sources and data-use boundary

Primary source:

- Kaggle dataset: `supergus/multistage-continuousflow-manufacturing-process`
- Source page description: real process data from an actual high-speed continuous production line near Detroit, Michigan; one production run spanning several hours.
- Sampling: 1 Hz.
- Structure: Stage 1 uses three parallel machines and a combiner; Stage 2 uses two serial machines; 15 output locations are measured after each stage.
- Kaggle data-file label: `© Original Authors`.

Because the source does not advertise a permissive redistribution license, this repository does not commit the raw CSV or source notes. `scripts/download_data.py` downloads the public source through KaggleHub into `data/raw/`, and `.gitignore` keeps those files untracked.

Only aggregate benchmark outputs are committed. The project does not copy Kaggle notebook code or claim third-party benchmark results as its own.
