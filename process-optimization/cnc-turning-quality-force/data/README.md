# Data

The raw source is intentionally downloaded at use time from the Kaggle dataset page documented in `SOURCES.md`.

Expected files after download:

```text
data/raw/
├── Exp1.csv
├── Exp2.csv
├── Prep.csv
└── manifest.json
```

`Exp1.csv` and `Exp2.csv` are planned machining experiments and are used by the benchmark. `Prep.csv` describes the tool-wear preparation phase and is preserved for provenance but is not mixed into the planned-experiment benchmark.

The source license is CC BY-NC-SA 4.0. Keeping the raw files source-downloaded avoids turning this repository into a redistribution mirror and keeps attribution/terms explicit at the point of acquisition.
