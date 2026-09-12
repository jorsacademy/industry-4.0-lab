# Sources and provenance

## Canonical dataset

UCI Machine Learning Repository — **SECOM**

- Dataset page: https://archive.ics.uci.edu/dataset/179/secom
- DOI: https://doi.org/10.24432/C54305
- Canonical archive: https://archive.ics.uci.edu/static/public/179/secom.zip
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)

UCI describes the data as measurements from a semiconductor manufacturing process monitored through signals collected from sensors and/or process measurement points. Each example represents one production entity with a pass/fail yield label and associated date/time stamp.

The metadata page reports 1,567 examples, 104 fails and 591 attributes. The raw `secom.data` matrix itself loads as 590 numeric process-measurement columns; labels and timestamps are stored separately in `secom_labels.data`.

## User-supplied mirror

The project was discovered through the Kaggle mirror:

- https://www.kaggle.com/datasets/paresh2047/uci-semcom

The implementation downloads from the canonical UCI source rather than depending on the mirror.

## Attribution

McCann, M. & Johnston, A. (2008). **SECOM** [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54305

Because the process variables are anonymous, this project does not assign physical engineering meanings or causal root causes to selected columns.
