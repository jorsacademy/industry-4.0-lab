# Data

Raw files are downloaded from the public Kaggle dataset `yuriykatser/industrial-data-from-the-arc-furnace` by `scripts/download_data.py`.

The source is a set of event/process tables keyed by `HEATID`. Raw CSV files are intentionally not committed; the downloader writes `data/raw/manifest.json` with file hashes for reproducibility.

Core tables used by the current leakage-safe temperature benchmark:

- `eaf_temp.csv` — repeated temperature and oxidation measurements;
- `eaf_transformer.csv` — transformer-stage operation;
- `basket_charged.csv` — initial material charging;
- `eaf_added_materials.csv` — additional furnace charging.

The source archive also contains higher-rate oxygen/gas and injected-carbon streams (`eaf_gaslance_mat.csv`, `inj_mat.csv`) plus chemistry and ladle-stage tables. They are retained by the downloader but are not used by the current v1 benchmark; adding them requires the same snapshot-time cutoff and a scalable streaming aggregation path.

License: MIT, as stated on the source dataset page.
