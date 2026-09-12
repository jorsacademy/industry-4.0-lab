from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

FEATURE_RE = re.compile(r"^L(?P<line>\d+)_S(?P<station>\d+)_(?P<kind>[FD])(?P<index>\d+)$")


@dataclass(frozen=True, order=True)
class StationKey:
    line: int
    station: int

    @property
    def label(self) -> str:
        return f"L{self.line}_S{self.station}"


def parse_feature_name(name: str) -> tuple[StationKey, str, int] | None:
    match = FEATURE_RE.match(name)
    if not match:
        return None
    return (
        StationKey(int(match.group("line")), int(match.group("station"))),
        match.group("kind"),
        int(match.group("index")),
    )


def group_columns_by_station(columns: Iterable[str], kind: str | None = None) -> dict[StationKey, list[str]]:
    grouped: dict[StationKey, list[str]] = defaultdict(list)
    for column in columns:
        parsed = parse_feature_name(column)
        if parsed is None:
            continue
        station, parsed_kind, _ = parsed
        if kind is not None and parsed_kind != kind:
            continue
        grouped[station].append(column)
    return {station: sorted(cols) for station, cols in sorted(grouped.items())}
