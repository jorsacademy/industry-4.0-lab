# Data

Raw files are downloaded from the public Kaggle dataset `yuriykatser/industrial-data-from-the-arc-furnace` by `scripts/download_data.py`.

The project uses the source tables as event logs keyed by `HEATID`. Raw CSV files are intentionally not committed; the downloader writes `data/raw/manifest.json` with file hashes for reproducibility.

Primary tables used by the benchmark:

- `eaf_temp.csv` — repeated temperature and oxidation measurements;
- `eaf_transformer.csv` — transformer-stage operation;
- `eaf_gaslance_mat.csv` — oxygen/gas cumulative usage and flows;
- `inj_mat.csv` — injected-carbon cumulative usage and flow;
- `basket_charged.csv` — initial material charging;
- `eaf_added_materials.csv` — additional furnace charging.

License: MIT, as stated on the source dataset page.
