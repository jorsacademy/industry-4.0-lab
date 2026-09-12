# Sources and data terms

## Primary data

Bosch Production Line Performance, Kaggle competition:

- https://www.kaggle.com/c/bosch-production-line-performance/data

Kaggle describes the data as measurements of parts moving through Bosch production lines. Feature names preserve production-line and station identifiers. The training target is `Response`, where `1` denotes a part that failed quality control.

The competition data page lists seven files totaling about 729 MB and marks the license as **Subject to Competition Rules**. For that reason, this repository does not redistribute the Bosch raw CSV files or derivatives that could substitute for obtaining the competition data through Kaggle.

Users must access the competition with their own Kaggle account, accept any applicable competition terms, and download through Kaggle's authorized mechanism.

## Dataset scale used as integrity checks

Public competition documentation reports:

- 1,183,747 training parts;
- 6,879 positive `Response=1` parts, approximately 0.58%;
- 968 numeric measurement columns plus `Id` and `Response` in `train_numeric.csv`;
- 1,156 date/timing columns plus `Id` in `train_date.csv`;
- 2,140 categorical columns plus `Id` in `train_categorical.csv`.

The project validates the raw schema locally but does not commit the competition data.

## Methodological context

The project is an independent implementation. No Kaggle notebook or third-party repository source code is copied. The design emphasizes station-aware sparse feature engineering, strict chronological evaluation, rare-event metrics, probability calibration, and early-warning trade-offs.
