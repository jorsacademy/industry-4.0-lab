from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dmdtrl import final_campaign as fc


def load_named_ppo_policy(model_path: Path, training_seed: int):
    """Load one frozen PPO member and give it a stable analysis-only policy name."""
    from dmdtrl.policies import load_ppo_policy

    policy = load_ppo_policy(model_path)
    policy.policy_name = fc.ppo_policy_name(training_seed)
    return policy


def run_frozen_campaign(  # pragma: no cover - exercised by final workflow integration
    design: dict[str, Any],
    ppo_freeze: dict[str, Any],
    cpsat_freeze: dict[str, Any],
    models_root: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    from dmdtrl import experiments
    from dmdtrl.env import EnvConfig
    from dmdtrl.or_experiments import evaluate_cpsat_policy
    from dmdtrl.or_policy import CPSATConfig, RollingHorizonCPSATPolicy
    from dmdtrl.research import fixed_policies
    from dmdtrl.scenarios import SCENARIO_REGISTRY

    verified_models = fc.verify_frozen_models(models_root, ppo_freeze)
    training_seeds = [int(seed) for seed in design["ppo_training_seeds"]]

    policies = list(fixed_policies())
    policies.extend(
        load_named_ppo_policy(verified_models[training_seed], training_seed)
        for training_seed in training_seeds
    )

    cpsat = RollingHorizonCPSATPolicy(
        CPSATConfig(
            max_jobs=int(cpsat_freeze["cpsat_horizon"]),
            time_limit_s=float(cpsat_freeze["solver_seconds"]),
        )
    )

    plans: list[tuple[str, list[int], EnvConfig]] = [
        (
            "nominal",
            list(
                range(
                    int(design["nominal"]["seed_start"]),
                    int(design["nominal"]["seed_start"])
                    + int(design["nominal"]["seed_count"]),
                )
            ),
            EnvConfig(),
        )
    ]
    for scenario_name in design["stress"]["scenarios"]:
        if scenario_name not in SCENARIO_REGISTRY:
            raise ValueError(f"unknown frozen stress scenario: {scenario_name}")
        plans.append(
            (
                scenario_name,
                list(
                    range(
                        int(design["stress"]["seed_start"]),
                        int(design["stress"]["seed_start"])
                        + int(design["stress"]["seed_count"]),
                    )
                ),
                SCENARIO_REGISTRY[scenario_name].apply(EnvConfig()),
            )
        )

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    dispersion_rows: list[dict[str, Any]] = []
    primary_comparisons: list[dict[str, Any]] = []
    per_model_comparisons: list[dict[str, Any]] = []

    for scenario_index, (scenario, seeds, env_config) in enumerate(plans):
        executable_rows = experiments.evaluate_policies(policies, seeds, env_config)
        executable_rows.extend(evaluate_cpsat_policy(cpsat, seeds, env_config))
        annotated = fc._annotate_policy_rows(
            executable_rows,
            scenario=scenario,
            training_seeds=training_seeds,
        )
        aggregate_rows = fc.average_ppo_by_environment_seed(
            annotated,
            training_seeds,
            scenario=scenario,
        )
        scenario_rows = annotated + aggregate_rows
        raw_rows.extend(scenario_rows)

        summaries = experiments.summarize_runs(
            scenario_rows,
            n_bootstrap=int(design["bootstrap"]),
            seed=70_000 + scenario_index * 10_000,
        )
        summary_rows.extend({"scenario": scenario, **summary} for summary in summaries)
        dispersion_rows.append(
            fc.ppo_training_seed_dispersion(annotated, training_seeds, scenario=scenario)
        )
        primary, secondary = fc._scenario_comparisons(
            scenario_rows,
            training_seeds,
            n_bootstrap=int(design["bootstrap"]),
            n_permutations=int(design["permutations"]),
            seed_offset=80_000 + scenario_index * 20_000,
        )
        primary_comparisons.extend(primary)
        per_model_comparisons.extend(secondary)

    fc.apply_primary_holm(primary_comparisons)
    return raw_rows, summary_rows, dispersion_rows, primary_comparisons, per_model_comparisons


def write_outputs(  # pragma: no cover - exercised by final workflow integration
    outputs: tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ],
    *,
    output_dir: Path,
    design: dict[str, Any],
    ppo_freeze: dict[str, Any],
    cpsat_freeze: dict[str, Any],
) -> None:
    raw, summaries, dispersion, primary, per_model = outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    fc.write_csv(raw, output_dir / "final_runs.csv")
    fc.write_csv(summaries, output_dir / "final_policy_summary.csv")
    fc.write_csv(dispersion, output_dir / "ppo_training_seed_dispersion.csv")
    fc.write_csv(primary, output_dir / "primary_comparisons.csv")
    fc.write_csv(per_model, output_dir / "ppo_per_model_comparisons.csv")
    (output_dir / "final_campaign_manifest.json").write_text(
        json.dumps(fc.build_manifest(design, ppo_freeze, cpsat_freeze), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:  # pragma: no cover - exercised by GitHub Actions
    parser = argparse.ArgumentParser(description="Execute the frozen final comparative campaign.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ppo-freeze", type=Path, required=True)
    parser.add_argument("--cpsat-freeze", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        ppo_freeze = fc.read_json(args.ppo_freeze)
        cpsat_freeze = fc.read_json(args.cpsat_freeze)
        design = fc.validate_design(fc.read_json(args.config), ppo_freeze, cpsat_freeze)
        outputs = run_frozen_campaign(design, ppo_freeze, cpsat_freeze, args.models_root)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        parser.error(str(exc))

    write_outputs(
        outputs,
        output_dir=args.output_dir,
        design=design,
        ppo_freeze=ppo_freeze,
        cpsat_freeze=cpsat_freeze,
    )

    _, summaries, _, _, _ = outputs
    print("Final weighted-tardiness leaders by scenario (descriptive only):")
    for scenario in ["nominal", *design["stress"]["scenarios"]]:
        candidates = [
            row
            for row in summaries
            if row["scenario"] == scenario
            and row["policy"]
            in {fc.PPO_AGGREGATE_POLICY, "CP_SAT_RH", fc.PRIMARY_FIXED_BASELINE}
        ]
        leader = min(candidates, key=lambda row: float(row["weighted_tardiness_mean"]))
        print(
            f"  {scenario:<20} {leader['policy']:<24} "
            f"WTT={float(leader['weighted_tardiness_mean']):.3f}"
        )


if __name__ == "__main__":
    main()
