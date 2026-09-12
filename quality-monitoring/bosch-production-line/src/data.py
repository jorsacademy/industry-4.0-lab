from __future__ import annotations

from itertools import zip_longest
from pathlib import Path
from typing import Iterator

import pandas as pd


def raw_paths(raw_dir: Path, numeric_name: str, date_name: str) -> tuple[Path, Path]:
    numeric_path = raw_dir / numeric_name
    date_path = raw_dir / date_name
    if not numeric_path.exists() or not date_path.exists():
        missing = [str(p) for p in (numeric_path, date_path) if not p.exists()]
        raise FileNotFoundError("Missing Bosch raw files: " + ", ".join(missing))
    return numeric_path, date_path


def iter_paired_chunks(
    numeric_path: Path,
    date_path: Path,
    *,
    chunk_size: int,
    max_rows: int | None = None,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    numeric_iter = pd.read_csv(numeric_path, chunksize=chunk_size)
    date_iter = pd.read_csv(date_path, chunksize=chunk_size)
    emitted = 0

    for numeric_chunk, date_chunk in zip_longest(numeric_iter, date_iter):
        if numeric_chunk is None or date_chunk is None:
            raise ValueError("Numeric and date files have different numbers of chunks")
        if max_rows is not None:
            remaining = max_rows - emitted
            if remaining <= 0:
                break
            if len(numeric_chunk) > remaining:
                numeric_chunk = numeric_chunk.iloc[:remaining].copy()
                date_chunk = date_chunk.iloc[:remaining].copy()
        if len(numeric_chunk) != len(date_chunk):
            raise ValueError("Numeric/date chunk lengths differ")
        if not numeric_chunk["Id"].equals(date_chunk["Id"]):
            raise ValueError("Numeric/date rows are not aligned by Id")
        emitted += len(numeric_chunk)
        yield numeric_chunk, date_chunk
        if max_rows is not None and emitted >= max_rows:
            break


def read_feature_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing feature table: {path}")
    return pd.read_parquet(path)
