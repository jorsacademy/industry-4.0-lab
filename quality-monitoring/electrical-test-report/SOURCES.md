# Sources and provenance

- Kaggle dataset: `joshipranjal5/electrical-test-report`
  - https://www.kaggle.com/datasets/joshipranjal5/electrical-test-report
  - Original file referenced by downstream research: `Electrical_Test_Report.csv`
- Independent dataset characterization:
  - Luling Duan and Pan Zhang, "A Sampling-Based Inspection and Cost Optimization Model for Electronic Assembly Quality Control," *Journal of Manufacturing and Materials Processing*, 2026, 10(5), 170.
  - https://doi.org/10.3390/jmmp10050170

The paper reports 80,000 cleaned records from 866 lots, 19 anonymized electrical measurement fields (`F1..F19`), 1.335% overall defective rate, and a strongly right-skewed lot-level defect distribution. The project uses those published figures only as integrity checks; predictive models use the downloaded source records themselves.
