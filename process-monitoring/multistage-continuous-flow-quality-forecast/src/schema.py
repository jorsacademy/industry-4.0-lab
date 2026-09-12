from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

OUTPUT_RE = re.compile(r"^Stage(?P<stage>[12])\.Output\.Measurement(?P<idx>\d+)\.U\.(?P<kind>Actual|Setpoint)$")


@dataclass(frozen=True)
class OutputColumn:
    stage: int
    index: int
    kind: str
    name: str


def parse_output_column(name: str) -> OutputColumn | None:
    match = OUTPUT_RE.match(str(name))
    if not match:
        return None
    return OutputColumn(
        stage=int(match.group("stage")),
        index=int(match.group("idx")),
        kind=match.group("kind").lower(),
        name=str(name),
    )


def output_columns(columns: Iterable[str], stage: int, kind: str) -> list[str]:
    kind = kind.lower()
    parsed = [x for c in columns if (x := parse_output_column(str(c))) is not None]
    selected = [x for x in parsed if x.stage == stage and x.kind == kind]
    return [x.name for x in sorted(selected, key=lambda x: x.index)]


def process_feature_columns(frame: pd.DataFrame) -> list[str]:
    stage2_actual = set(output_columns(frame.columns, stage=2, kind="actual"))
    excluded = {"time_stamp", *stage2_actual}
    return [
        c
        for c in frame.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])
    ]
