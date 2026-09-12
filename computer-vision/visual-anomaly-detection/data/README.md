# Local data

Raw images are not tracked in Git.

Expected layout:

```text
data/raw/
├── train/
│   └── normal/
├── test/
│   ├── normal/
│   └── anomaly/
```

Use only conforming parts under `train/normal`. The `test/anomaly` directory is reserved for final evaluation and must not be used to select the model threshold.
