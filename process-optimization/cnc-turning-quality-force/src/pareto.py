from __future__ import annotations

import numpy as np
import pandas as pd


def pareto_mask(
    frame: pd.DataFrame,
    *,
    minimize: list[str],
    maximize: list[str],
) -> np.ndarray:
    values_min = frame[minimize].to_numpy(dtype=float) if minimize else np.empty((len(frame), 0))
    values_max = frame[maximize].to_numpy(dtype=float) if maximize else np.empty((len(frame), 0))
    keep = np.ones(len(frame), dtype=bool)
    for i in range(len(frame)):
        if not keep[i]:
            continue
        for j in range(len(frame)):
            if i == j:
                continue
            no_worse_min = np.all(values_min[j] <= values_min[i]) if minimize else True
            no_worse_max = np.all(values_max[j] >= values_max[i]) if maximize else True
            strictly_better = (
                (np.any(values_min[j] < values_min[i]) if minimize else False)
                or (np.any(values_max[j] > values_max[i]) if maximize else False)
            )
            if no_worse_min and no_worse_max and strictly_better:
                keep[i] = False
                break
    return keep


def score_supported_recipes(
    run_level: pd.DataFrame,
    roughness_model,
    force_model,
    *,
    setpoint_features: list[str],
    by_tool_condition: bool = True,
) -> pd.DataFrame:
    recipes = (
        run_level[setpoint_features]
        .drop_duplicates()
        .sort_values(setpoint_features)
        .reset_index(drop=True)
    )
    recipes["predicted_Ra"] = roughness_model.predict(recipes[setpoint_features])
    recipes["predicted_F"] = force_model.predict(recipes[setpoint_features])
    recipes["mrr_proxy"] = recipes["ap"] * recipes["f"] * recipes["vc"]
    recipes["pareto_efficient"] = False

    if by_tool_condition and "TCond" in recipes.columns:
        for _, idx in recipes.groupby("TCond", dropna=False).groups.items():
            subset = recipes.loc[idx]
            recipes.loc[idx, "pareto_efficient"] = pareto_mask(
                subset,
                minimize=["predicted_Ra", "predicted_F"],
                maximize=["mrr_proxy"],
            )
    else:
        recipes["pareto_efficient"] = pareto_mask(
            recipes,
            minimize=["predicted_Ra", "predicted_F"],
            maximize=["mrr_proxy"],
        )
    return recipes.sort_values(["TCond", "pareto_efficient", "mrr_proxy"], ascending=[True, False, False])
