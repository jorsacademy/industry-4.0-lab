# Data

Canonical source: UCI SECOM semiconductor manufacturing dataset.

Run:

```bash
python scripts/download_data.py
python scripts/validate_data.py
```

The downloader retrieves the UCI archive, preserves `secom.data`, `secom_labels.data`, and `secom.names`, and writes a SHA-256 manifest.

The dataset is licensed CC BY 4.0 by UCI. Raw files are kept source-downloaded in this monorepo workflow rather than duplicated in git so that provenance remains explicit and benchmark jobs always validate the canonical distribution.

Expected raw structure:

```text
data/raw/
├── secom.data
├── secom_labels.data
├── secom.names
└── manifest.json
```

`secom.data` loads as 590 anonymous numeric process variables. `secom_labels.data` supplies the pass/fail label and timestamp.
