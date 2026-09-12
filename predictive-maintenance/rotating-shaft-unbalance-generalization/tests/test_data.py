import numpy as np
import pandas as pd

from src.data import REQUIRED_COLUMNS, iter_nonoverlap_windows


def _chunk(start, size):
    values = np.arange(start, start + size, dtype=float)
    frame = pd.DataFrame({c: values for c in REQUIRED_COLUMNS})
    return frame


def test_chunked_windowing_preserves_global_alignment():
    chunks = iter([_chunk(0, 7), _chunk(7, 6), _chunk(13, 7)])
    windows = list(iter_nonoverlap_windows(chunks, window_samples=4, hop_samples=6, skip_samples=2))
    starts = [start for start, _ in windows]
    assert starts == [2, 8, 14]
    assert windows[0][1][0, 0] == 2
    assert windows[1][1][-1, 0] == 11


def test_overlapping_hop_is_rejected():
    chunks = iter([_chunk(0, 10)])
    try:
        list(iter_nonoverlap_windows(chunks, window_samples=5, hop_samples=4, skip_samples=0))
    except ValueError as exc:
        assert "hop_samples" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
