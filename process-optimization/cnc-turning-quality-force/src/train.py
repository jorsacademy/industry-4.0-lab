from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

from .data import available_force_columns, load_run_level
from .modeling import choose_best, evaluate_task, feature_importance_table, fit_selected
from .pareto import score_supported_recipes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grouped CNC turning response-model benchmark.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    project_root = cfg_path.resolve().parent
    data_cfg = cfg["data"]
    model_cfg = cfg["modeling"]
    report_cfg = cfg["reports"]

    raw_dir = project_root / data_cfg["raw_dir"]
    frame = load_run_level(raw_dir, data_cfg["exp1_file"], data_cfg["exp2_file"])

    setpoint_features = ["ap", "vc", "f", "TCond"]
    force_columns = available_force_columns(frame)
    if "F" not in force_columns:
        raise ValueError("Resultant force F is required for the process-load surrogate")
    force_assisted_features = setpoint_features + [c for c in ["Fx", "Fy", "Fz", "F"] if c in force_columns]

    candidates = list(model_cfg.get("candidates", ["ridge", "random_forest", "extra_trees"]))
    random_state = int(model_cfg.get("random_state", 42))
    n_splits = int(model_cfg.get("cv_splits", 5))
    roughness_target = str(model_cfg.get("roughness_target", "Ra_mean"))
    force_target = str(model_cfg.get("force_target", "F"))

    task_specs = {
        "roughness_setpoint": (setpoint_features, roughness_target),
        "roughness_force_assisted": (force_assisted_features, roughness_target),
        "force_setpoint": (setpoint_features, force_target),
    }

    all_results = []
    best_by_task = {}
    for task, (features, target) in task_specs.items():
        results = evaluate_task(
            frame,
            task=task,
            features=features,
            target=target,
            candidates=candidates,
            random_state=random_state,
            n_splits=n_splits,
        )
        all_results.extend(results)
        best_by_task[task] = choose_best(results)

    reports = project_root / report_cfg.get("directory", "reports")
    artifacts = project_root / report_cfg.get("artifacts_directory", "artifacts")
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    for result in all_results:
        comparison_rows.append({
            "task": result.task,
            "model": result.model_name,
            "features": ",".join(result.features),
            **result.metrics,
        })
    comparison = pd.DataFrame(comparison_rows).sort_values(["task", "mae", "rmse"])
    comparison.to_csv(reports / "model_comparison.csv", index=False)

    fitted = {}
    importances = []
    oof = frame[["experiment", "Run", "run_key", "condition_key", "ap", "vc", "f", "TCond", "Ra_mean", "Ra_std", "F", "mrr_proxy"]].copy()
    for task, best in best_by_task.items():
        features, target = task_specs[task]
        usable = frame.dropna(subset=[target]).reset_index(drop=True)
        model = fit_selected(best.model_name, usable[features], usable[target], random_state=random_state)
        fitted[task] = {"pipeline": model, "features": features, "target": target, "model_name": best.model_name}
        importances.append(feature_importance_table(model, features, task))

        if len(usable) != len(frame):
            raise ValueError(f"Task {task} dropped rows due to missing target; explicit handling is required")
        oof[f"pred_{task}"] = best.oof_prediction
        oof[f"residual_{task}"] = frame[target].to_numpy(dtype=float) - best.oof_prediction

    oof.to_csv(reports / "oof_predictions.csv", index=False)
    pd.concat(importances, ignore_index=True).to_csv(reports / "feature_importance.csv", index=False)
    frame.to_csv(reports / "run_level_summary.csv", index=False)

    pareto = score_supported_recipes(
        frame,
        fitted["roughness_setpoint"]["pipeline"],
        fitted["force_setpoint"]["pipeline"],
        setpoint_features=setpoint_features,
        by_tool_condition=bool(cfg.get("optimization", {}).get("pareto_by_tool_condition", True)),
    )
    pareto.to_csv(reports / "pareto_candidates.csv", index=False)

    setpoint_mae = best_by_task["roughness_setpoint"].metrics["mae"]
    assisted_mae = best_by_task["roughness_force_assisted"].metrics["mae"]
    improvement = float((setpoint_mae - assisted_mae) / setpoint_mae) if setpoint_mae else float("nan")

    metrics = {
        "source_rows": int(frame["n_positions"].sum()),
        "machining_runs": int(len(frame)),
        "exp1_runs": int((frame["experiment"] == "Exp1").sum()),
        "exp2_runs": int((frame["experiment"] == "Exp2").sum()),
        "unique_process_conditions": int(frame["condition_key"].nunique()),
        "validation_unit": "process-condition grouped cross-validation; replicate runs remain in the same fold",
        "selected": {
            task: {"model": result.model_name, "features": result.features, **result.metrics}
            for task, result in best_by_task.items()
        },
        "force_assisted_roughness_mae_relative_improvement": improvement,
        "pareto_candidates": int(len(pareto)),
        "pareto_efficient_candidates": int(pareto["pareto_efficient"].sum()),
        "interpretation_note": "Pareto candidates are restricted to experimentally observed recipes; they are decision-support candidates, not universal production optima.",
    }
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump({
        "models": fitted,
        "setpoint_features": setpoint_features,
        "force_columns": force_columns,
    }, artifacts / "cnc_turning_surrogates.joblib")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
