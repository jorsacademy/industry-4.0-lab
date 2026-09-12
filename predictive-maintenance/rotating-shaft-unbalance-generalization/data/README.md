# Data handling

The project uses the canonical Fordatis/Fraunhofer raw CSV archive from `fordatis/151.2` under CC BY 4.0.

`python scripts/download_data.py` writes the archive to `data/raw/` and creates `data/raw/manifest.json` with the source URL, byte count and SHA-256 digest. The raw archive is ignored by Git.

The benchmark streams `0D.csv ... 4D.csv` and `0E.csv ... 4E.csv` directly from the ZIP. It does not extract the multi-gigabyte CSV members to the working tree.

The `D` and `E` recordings are kept separate by design. `E` is never used for model selection.
