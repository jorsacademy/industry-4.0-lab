import pandas as pd

from src.data import (
    ACTIVE_CONTROL_COLS,
    OUTPUT_COLS,
    THICKNESS_COLS,
    active_production_mask,
    interruption_onsets,
    predictor_columns,
    segment_ids,
    upcoming_event_target,
)


def _frame():
    rows = []
    for _ in range(5):
        row = {c: 1.0 for c in THICKNESS_COLS + OUTPUT_COLS}
        row.update(
            {
                "ST112_VARAbzug_1_IstEin": 1.0,
                "ST112_VARAbzug_1_IstZu": 1.0,
                "ST112_VARAbzug_1_SollSpeed": 20.0,
                "ST113_VARLmpRun": 1.0,
                "ST114_VARLmpRun": 0.0,
                "safe_sensor": 3.0,
                "Datum": "unused",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_active_state_requires_physical_production_signals():
    frame = _frame()
    frame.loc[1, THICKNESS_COLS] = 0.0
    frame.loc[2, "ST112_VARAbzug_1_SollSpeed"] = 5.0
    frame.loc[3, ["ST113_VARLmpRun", "ST114_VARLmpRun"]] = 0.0
    mask = active_production_mask(frame)
    assert mask.tolist() == [True, False, False, False, True]


def test_interruption_onset_requires_previous_active_row_and_normal_gap():
    frame = _frame()
    frame.loc[1, THICKNESS_COLS] = 0.0
    frame.loc[2, THICKNESS_COLS] = 0.0
    ts = pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 00:02", "2024-01-01 00:04", "2024-01-01 00:06", "2024-01-01 00:08"]
    ).to_series(index=frame.index)
    active = active_production_mask(frame)
    event = interruption_onsets(frame, ts, active, onset_max_gap_minutes=3)
    assert event.tolist() == [False, True, False, False, False]


def test_upcoming_target_does_not_cross_segment_gap():
    ts = pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 00:02", "2024-01-01 00:04", "2024-01-01 01:00", "2024-01-01 01:02"]
    ).to_series(index=range(5))
    segments = segment_ids(ts, segment_gap_minutes=5)
    active = pd.Series([True, True, False, True, False])
    event = pd.Series([False, False, True, False, True])
    target, lead = upcoming_event_target(ts, segments, active, event, warning_horizon_minutes=20)
    assert target.tolist() == [1, 1, 0, 1, 0]
    assert lead.iloc[1] == 2.0


def test_direct_target_state_columns_are_excluded_from_predictors():
    frame = _frame()
    columns = predictor_columns(frame)
    assert "safe_sensor" in columns
    for col in THICKNESS_COLS + OUTPUT_COLS + ACTIVE_CONTROL_COLS:
        assert col not in columns
