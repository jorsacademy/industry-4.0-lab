# Data

The raw Bosch competition files are intentionally **not tracked** because the dataset is governed by Kaggle competition-specific terms rather than a general redistribution license.

After accepting the competition rules with your own Kaggle account, run:

```bash
python scripts/download_data.py
```

Expected raw files:

```text
data/raw/
├── train_numeric.csv
├── train_date.csv
├── train_categorical.csv
├── test_numeric.csv
├── test_date.csv
├── test_categorical.csv
└── sample_submission.csv
```

The default modeling path uses `train_numeric.csv` and `train_date.csv`. Categorical features are excluded from the first production baseline because they are extremely sparse; the station route and timing structure is retained through date-feature aggregation.

Generated station-level feature tables are written to `data/processed/` and are ignored by Git.

## Why date features matter

A Bosch feature such as `L3_S36_F3939` is a measurement on line 3, station 36. Date features use the same line/station convention and provide measurement times. The feature builder uses them to recover per-part process start/end time, duration, station presence, station timing, and an approximate early-process prefix.

## Early-warning prefixes

`src.build_features` accepts a prefix fraction. At `0.25`, for example, numeric station aggregates are exposed only for stations whose observed station timestamp falls within the first quarter of that part's measured process duration. This supports a realistic question: how early can a high-risk part be flagged before its route is complete?
