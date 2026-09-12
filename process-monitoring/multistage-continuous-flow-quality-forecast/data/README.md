# Data

Run:

```bash
python scripts/download_data.py
```

The downloader uses KaggleHub to obtain the public dataset `supergus/multistage-continuousflow-manufacturing-process` and requests:

- `continuous_factory_process.csv`
- `notes_on_dataset.txt`

The source data are kept under `data/raw/` and are intentionally ignored by Git because the Kaggle page labels the data files `© Original Authors` rather than providing a permissive redistribution license.

The benchmark expects a 1 Hz time series with 14,088 rows and 116 columns. `scripts/validate_data.py` checks the expected Stage-1/Stage-2 output structure before training.
