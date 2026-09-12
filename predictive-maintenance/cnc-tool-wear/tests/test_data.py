import pandas as pd

from src.data import clean_experiment, known_artifact_mask


def test_known_artifact_mask_flags_documented_states():
    frame = pd.DataFrame(
        {
            "M1_CURRENT_FEEDRATE": [10, 50, 12, 12],
            "X1_ActualPosition": [0, 1, 198, 3],
            "M1_CURRENT_PROGRAM_NUMBER": [0, 0, 0, 7],
            "Machining_Process": ["A", "A", "A", "A"],
        }
    )
    assert known_artifact_mask(frame).tolist() == [False, True, True, True]


def test_clean_experiment_can_drop_artifact_rows():
    frame = pd.DataFrame(
        {
            "M1_CURRENT_FEEDRATE": [10, 50, 12],
            "X1_ActualPosition": [0.0, 1.0, 2.0],
            "M1_CURRENT_PROGRAM_NUMBER": [0, 0, 0],
            "Machining_Process": ["A", "A", "A"],
        }
    )
    cleaned = clean_experiment(frame, drop_known_artifacts=True)
    assert len(cleaned) == 2
    assert cleaned["_known_artifact"].sum() == 0
