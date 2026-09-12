import pandas as pd

from src.schema import output_columns, process_feature_columns


def test_output_columns_are_sorted_by_measurement_index():
    columns = [
        "Stage2.Output.Measurement10.U.Actual",
        "Stage2.Output.Measurement2.U.Actual",
        "Stage2.Output.Measurement1.U.Actual",
        "other",
    ]
    assert output_columns(columns, 2, "actual") == [
        "Stage2.Output.Measurement1.U.Actual",
        "Stage2.Output.Measurement2.U.Actual",
        "Stage2.Output.Measurement10.U.Actual",
    ]


def test_stage2_actuals_are_excluded_from_process_features():
    frame = pd.DataFrame({
        "time_stamp": ["2024-01-01"],
        "Machine1.MotorRPM.C.Actual": [10.0],
        "Stage1.Output.Measurement0.U.Actual": [1.0],
        "Stage2.Output.Measurement0.U.Setpoint": [2.0],
        "Stage2.Output.Measurement0.U.Actual": [3.0],
    })
    features = process_feature_columns(frame)
    assert "Machine1.MotorRPM.C.Actual" in features
    assert "Stage1.Output.Measurement0.U.Actual" in features
    assert "Stage2.Output.Measurement0.U.Setpoint" in features
    assert "Stage2.Output.Measurement0.U.Actual" not in features
