# Data

Source: Kaggle dataset **Electrical Test Report** by `joshipranjal5`.

Dataset page: https://www.kaggle.com/datasets/joshipranjal5/electrical-test-report

The source file is `Electrical_Test_Report.csv`. Independent published analysis of this dataset reports that it is UTF-16 encoded and contains repeated header rows inside the body. After removing 15 repeated headers, the data contain 80,000 electrical test records from 866 production lots dated from 4 October 2019 through 20 December 2019. The final `Result` field contains 78,932 passing records and 1,068 defective records; electrical measurements are anonymized as `F1` through `F19`.

## Why raw data is not committed here

The dataset is downloaded from its source at use time because its redistribution license was not independently verified while this project was assembled. Run:

```bash
python scripts/download_data.py
python scripts/validate_data.py
```

The downloader preserves the original CSV and writes a local SHA-256 manifest. Do not commit the downloaded CSV unless the source license is checked and explicitly permits redistribution.

## Cleaning policy

Only source-format artifacts are removed. Specifically, repeated header records are filtered, while valid production observations are retained. The original `LOT`, `Time`, `Result`, and `F1..F19` fields remain available; derived columns such as parsed timestamps and `is_defect` are created in memory.
