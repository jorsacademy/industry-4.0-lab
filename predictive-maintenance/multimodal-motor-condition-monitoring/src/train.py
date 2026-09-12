from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix

from .data import blocked_purged_folds, load_frequency_table, validate_frequency_table
from .modeling import candidate_models, evaluate_candidate


def _feature_sets(accel: list[str], audio: list[str]) -> dict[str, list[str]]:
    return {"vibration": accel, "audio": audio, "fusion": accel + audio}


def run(config_path: str | Path) -> dict[str, object]:
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = config_path.parent
    table = load_frequency_table(root / cfg["data"]["raw_dir"])
    df = table.frame.reset_index(drop=True)
    y = df["condition"].astype(str)

    eval_cfg = cfg["evaluation"]
    folds = blocked_purged_folds(df, n_splits=int(eval_cfg["n_splits"]), purge_windows=int(eval_cfg["purge_windows"]))

    comparisons: list[dict[str, object]] = []
    predictions: dict[tuple[str, str], object] = {}
    models = candidate_models(int(cfg["model"]["random_state"]))
    for modality, features in _feature_sets(table.acceleration_columns, table.audio_columns).items():
        X = df[features]
        for model_name, estimator in models.items():
            metrics, pred = evaluate_candidate(X, y, folds, estimator)
            comparisons.append({"modality": modality, "model": model_name, "n_features": len(features), **metrics})
            predictions[(modality, model_name)] = pred

    comparison = pd.DataFrame(comparisons).sort_values(
        ["macro_f1", "balanced_accuracy", "n_features"], ascending=[False, False, True]
    ).reset_index(drop=True)
    best = comparison.iloc[0]
    best_key = (str(best["modality"]), str(best["model"]))
    best_features = _feature_sets(table.acceleration_columns, table.audio_columns)[best_key[0]]
    best_pred = predictions[best_key]

    final_model = clone(models[best_key[1]])
    final_model.fit(df[best_features], y)

    reports_dir = root / cfg["output"]["reports_dir"]
    artifacts_dir = root / cfg["output"]["artifacts_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(reports_dir / "model_modality_comparison.csv", index=False)
    pd.DataFrame({
        "source_row": df["source_row"], "condition": y, "window_index": df["window_index"], "oof_prediction": best_pred,
    }).to_csv(reports_dir / "blocked_oof_predictions.csv", index=False)

    labels = sorted(y.unique())
    cm = confusion_matrix(y, best_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(reports_dir / "confusion_matrix.csv")
    pd.DataFrame(classification_report(y, best_pred, labels=labels, output_dict=True, zero_division=0)).transpose().to_csv(
        reports_dir / "class_report.csv"
    )

    metrics = {
        "source": validate_frequency_table(table),
        "evaluation": {
            "protocol": "same-session blocked cross-validation with a purge gap around each held-out time block",
            "n_splits": int(eval_cfg["n_splits"]),
            "purge_windows": int(eval_cfg["purge_windows"]),
            "window_overlap_fraction": float(eval_cfg["window_overlap_fraction"]),
            "independent_session_generalization": False,
        },
        "selected": {
            "modality": best_key[0], "model": best_key[1], "n_features": int(best["n_features"]),
            "macro_f1": float(best["macro_f1"]), "macro_f1_std": float(best["macro_f1_std"]),
            "balanced_accuracy": float(best["balanced_accuracy"]), "accuracy": float(best["accuracy"]),
        },
        "interpretation_note": (
            "The source provides one short recording per operating condition. Blocked/purged evaluation avoids direct overlap leakage, "
            "but it remains a same-session benchmark and is not evidence of generalization to new machines, days, microphones, or sensor mountings."
        ),
    }
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump({"model": final_model, "features": best_features, "labels": labels, "metrics": metrics}, artifacts_dir / "condition_model.joblib")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
