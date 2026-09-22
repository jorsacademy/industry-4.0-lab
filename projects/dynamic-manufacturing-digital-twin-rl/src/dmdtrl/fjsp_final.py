from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from dmdtrl.fjsp_evaluate import compare_candidate, instance_fingerprint, summarize_panel, write_csv
from dmdtrl.fjsp_generator import generate_fjsp_instance
from dmdtrl.fjsp_hh_campaign import build_env_config, build_generator_config, validate_cpsat_freeze
from dmdtrl.fjsp_hh_evaluate import _run_cpsat, _run_operator
from dmdtrl.fjsp_operators import OPERATOR_NAMES, FJSPOperator

CPSAT_POLICY = "ROLLING_HORIZON_CPSAT"
FINAL_SEED_START = 42_000
FINAL_SEED_END = 42_099


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def final_seeds(config: dict[str, Any]) -> list[int]:
    start = int(config["seed_start"])
    end = int(config["seed_end"])
    count = int(config["seed_count"])
    seeds = list(range(start, start + count))
    if not seeds or seeds[-1] != end:
        raise ValueError("final seed range does not match seed_count")
    if start != FINAL_SEED_START or end != FINAL_SEED_END or count != 100:
        raise ValueError("Phase-5 final block must remain exactly 42000-42099")
    return seeds


def validate_final_design(
    config: dict[str, Any],
    environment_design: dict[str, Any],
    cpsat_freeze: dict[str, Any],
    rl_decision: dict[str, Any],
) -> None:
    if config.get("status") != "predeclared_final_frozen_baseline_test":
        raise ValueError("final config must be predeclared_final_frozen_baseline_test")
    if config.get("purpose") != "project_completion_after_rl_validation_rejection":
        raise ValueError("unexpected final benchmark purpose")
    final_seeds(config)

    expected_policies = [*OPERATOR_NAMES, CPSAT_POLICY]
    if list(config.get("policies", [])) != expected_policies:
        raise ValueError("final policy panel drifted from the frozen controller set")
    if config.get("candidate") != CPSAT_POLICY:
        raise ValueError("frozen CP-SAT must be the final comparison candidate")
    if set(config.get("excluded_from_final", [])) != {"MASKABLE_PPO", "PPO_HYPER_HEURISTIC"}:
        raise ValueError("rejected RL controllers must stay excluded from final testing")

    authorization = config.get("authorization")
    if not isinstance(authorization, dict):
        raise ValueError("missing final authorization block")
    if authorization.get("scope") != "frozen_non_rl_controllers_only":
        raise ValueError("final authorization scope drifted")
    if authorization.get("selection_complete_before_final_access") is not True:
        raise ValueError("selection must be complete before final access")
    if authorization.get("retuning_after_final_access") is not False:
        raise ValueError("retuning after final access is prohibited")
    if authorization.get("future_rl_selection_on_42000_42099") is not False:
        raise ValueError("final block cannot be reused for future RL selection")

    if int(environment_design.get("final_test_seed_start", -1)) != FINAL_SEED_START:
        raise ValueError("environment design final seed start mismatch")
    if int(environment_design.get("final_test_seed_end", -1)) != FINAL_SEED_END:
        raise ValueError("environment design final seed end mismatch")
    if rl_decision.get("decision", {}).get("promote_to_phase5_final") is not False:
        raise ValueError("RL validation decision must remain rejected")
    if rl_decision.get("data_boundary", {}).get("final_test_used_for_selection") is not False:
        raise ValueError("RL selection must not have used final data")
    if rl_decision.get("data_boundary", {}).get("final_test_seed_start") != FINAL_SEED_START:
        raise ValueError("RL decision final seed start mismatch")
    if rl_decision.get("data_boundary", {}).get("final_test_seed_end") != FINAL_SEED_END:
        raise ValueError("RL decision final seed end mismatch")

    if cpsat_freeze.get("selected_policy") != "FJSP_CPSAT_H4_B100MS":
        raise ValueError("unexpected frozen FJSP CP-SAT policy")
    if cpsat_freeze.get("selection_data_boundary", {}).get("final_test_used_for_selection") is not False:
        raise ValueError("CP-SAT selection must not have used final data")


def evaluate_final_panel(
    config: dict[str, Any],
    environment_design: dict[str, Any],
    cpsat_freeze: dict[str, Any],
) -> list[dict[str, float | int | str]]:
    generator = build_generator_config(environment_design)
    env_config = build_env_config(environment_design)
    cpsat_config = validate_cpsat_freeze(cpsat_freeze, environment_design)

    rows: list[dict[str, float | int | str]] = []
    for seed in final_seeds(config):
        instance = generate_fjsp_instance(np.random.default_rng(seed), generator)
        fingerprint = instance_fingerprint(instance)
        results = [
            _run_operator(instance, operator, setup_time=env_config.default_setup_time)
            for operator in FJSPOperator
        ]
        results.append(
            _run_cpsat(
                instance,
                setup_time=env_config.default_setup_time,
                config=cpsat_config,
            )
        )
        for result in results:
            rows.append(
                {
                    "seed": seed,
                    "seed_regime": "final",
                    "instance_sha256": fingerprint,
                    **result,
                }
            )
    validate_final_rows(config, rows)
    return rows


def validate_final_rows(
    config: dict[str, Any], rows: list[dict[str, float | int | str]]
) -> None:
    seeds = set(final_seeds(config))
    policies = set(config["policies"])
    expected_rows = len(seeds) * len(policies)
    if len(rows) != expected_rows:
        raise ValueError("final panel contains duplicate or missing rows")

    by_policy: dict[str, set[int]] = defaultdict(set)
    fingerprints: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        policy = str(row["policy"])
        seed = int(row["seed"])
        if policy not in policies:
            raise ValueError(f"unexpected final policy {policy!r}")
        if seed not in seeds:
            raise ValueError("final row escaped the frozen seed block")
        if row.get("seed_regime") != "final":
            raise ValueError("final row has incorrect seed regime")
        by_policy[policy].add(seed)
        fingerprints[seed].add(str(row["instance_sha256"]))

    if set(by_policy) != policies:
        raise ValueError("final panel is missing frozen policies")
    if any(policy_seeds != seeds for policy_seeds in by_policy.values()):
        raise ValueError("final policy seed sets are incomplete")
    if set(fingerprints) != seeds or any(len(values) != 1 for values in fingerprints.values()):
        raise ValueError("final controllers did not share one canonical instance per seed")


def build_manifest(
    config: dict[str, Any],
    summary: list[dict[str, float | int | str]],
    comparisons: list[dict[str, float | int | str]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": 5,
        "status": "final_frozen_baseline_test_complete",
        "scope": "frozen_non_rl_controllers_only",
        "final_seed_start": FINAL_SEED_START,
        "final_seed_end": FINAL_SEED_END,
        "final_seed_count": 100,
        "candidate": CPSAT_POLICY,
        "excluded_rl_controllers": list(config["excluded_from_final"]),
        "summary": summary,
        "comparisons": comparisons,
        "no_retuning_after_final": True,
        "final_block_consumed": True,
        "git": {
            "sha": os.getenv("GITHUB_SHA"),
            "ref": os.getenv("GITHUB_REF"),
            "workflow": os.getenv("GITHUB_WORKFLOW"),
            "run_id": os.getenv("GITHUB_RUN_ID"),
        },
    }


def run_final_benchmark(
    *,
    config_path: str | Path,
    environment_design_path: str | Path,
    cpsat_freeze_path: str | Path,
    rl_decision_path: str | Path,
    output_root: str | Path,
    bootstrap: int,
    permutations: int,
) -> tuple[
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    dict[str, Any],
]:
    config = read_json(config_path)
    environment_design = read_json(environment_design_path)
    cpsat_freeze = read_json(cpsat_freeze_path)
    rl_decision = read_json(rl_decision_path)
    validate_final_design(config, environment_design, cpsat_freeze, rl_decision)

    rows = evaluate_final_panel(config, environment_design, cpsat_freeze)
    summary = summarize_panel(rows, bootstrap=bootstrap)
    comparisons = compare_candidate(
        rows,
        candidate=CPSAT_POLICY,
        metric="weighted_tardiness",
        bootstrap=bootstrap,
        permutations=permutations,
    )
    manifest = build_manifest(config, summary, comparisons)

    root = Path(output_root)
    write_csv(root / "fjsp_final_runs.csv", rows)
    write_csv(root / "fjsp_final_summary.csv", summary)
    write_csv(root / "fjsp_final_cpsat_comparisons.csv", comparisons)
    root.mkdir(parents=True, exist_ok=True)
    (root / "fjsp_final_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows, summary, comparisons, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the one-time Phase-5 final benchmark for frozen non-RL FJSP controllers."
    )
    parser.add_argument("--config", default="configs/fjsp_final_baseline_test.json")
    parser.add_argument("--environment-design", default="configs/fjsp_hh_validation_design.json")
    parser.add_argument("--cpsat-freeze", default="configs/fjsp_cpsat_validation_freeze.json")
    parser.add_argument("--rl-decision", default="configs/fjsp_hh_validation_decision.json")
    parser.add_argument("--output-root", default="results/fjsp_final")
    parser.add_argument("--bootstrap", type=int, default=5_000)
    parser.add_argument("--permutations", type=int, default=10_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, summary, comparisons, _ = run_final_benchmark(
        config_path=args.config,
        environment_design_path=args.environment_design,
        cpsat_freeze_path=args.cpsat_freeze,
        rl_decision_path=args.rl_decision,
        output_root=args.output_root,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
    )
    for rank, row in enumerate(summary, start=1):
        print(
            f"{rank}. {row['policy']}: WTT={float(row['weighted_tardiness_mean']):.3f} "
            f"latency={float(row['mean_decision_time_ms']):.3f} ms"
        )
    print("CP-SAT paired final comparisons:")
    for row in comparisons:
        print(
            f"  vs {row['baseline']}: improvement={float(row['mean_improvement']):.3f} "
            f"CI=[{float(row['ci_low']):.3f}, {float(row['ci_high']):.3f}]"
        )


if __name__ == "__main__":
    main()
