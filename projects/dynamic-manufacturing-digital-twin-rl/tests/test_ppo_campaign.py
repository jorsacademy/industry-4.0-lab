from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from dmdtrl.ppo_campaign import (
    aggregate_member_metrics,
    select_representative_member,
    validate_design,
    validate_member,
)


def design() -> dict:
    return {
        "algorithm": "PPO",
        "training_seeds": [101, 202, 303],
        "validation_seed_start": 10000,
        "validation_seed_count": 2,
        "final_test_seed_start": 20000,
        "stress_test_seed_start": 30000,
        "training_config": {
            "total_timesteps": 150000,
            "learning_rate": 0.0003,
            "n_steps": 1024,
            "batch_size": 128,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
            "hidden_units": 128,
            "device": "cpu",
            "verbose": 0,
        },
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_member(root: Path, seed: int, wtt: float, latency: float) -> None:
    member = root / f"seed_{seed}"
    member.mkdir(parents=True)
    write_csv(
        member / "validation_runs.csv",
        [
            {"policy": "PPO", "seed": 10000, "weighted_tardiness": wtt},
            {"policy": "PPO", "seed": 10001, "weighted_tardiness": wtt + 1},
        ],
    )
    write_csv(
        member / "validation_summary.csv",
        [
            {
                "policy": "PPO",
                "weighted_tardiness_mean": wtt,
                "weighted_tardiness_ci_low": wtt - 2,
                "weighted_tardiness_ci_high": wtt + 2,
                "mean_decision_time_ms_mean": latency,
                "mean_waiting_time_mean": 2.0,
                "on_time_rate_mean": 0.9,
                "makespan_mean": 100.0,
            }
        ],
    )
    manifest = {
        "algorithm": "PPO",
        "training_seed": seed,
        "training_seconds": 12.5,
        "training_config": {**design()["training_config"], "seed": seed},
    }
    (member / "training_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (member / "ppo_dispatcher.zip").write_bytes(f"model-{seed}".encode())


def test_validate_design_enforces_disjoint_seed_regimes() -> None:
    valid = validate_design(design())
    assert valid["training_seeds"] == [101, 202, 303]

    bad = design()
    bad["training_seeds"] = [101, 10000, 303]
    with pytest.raises(ValueError, match="below validation"):
        validate_design(bad)

    bad = design()
    bad["validation_seed_start"] = 19999
    bad["validation_seed_count"] = 2
    with pytest.raises(ValueError, match=r"\[10000, 20000\)"):
        validate_design(bad)


def test_validate_member_checks_complete_common_validation_seeds(tmp_path: Path) -> None:
    cfg = validate_design(design())
    create_member(tmp_path, 101, 80.0, 0.5)
    member, rows = validate_member(cfg, tmp_path, 101)
    assert member["training_seed"] == 101
    assert member["n_validation_seeds"] == 2
    assert len(rows) == 2
    assert all(row["training_seed"] == 101 for row in rows)

    write_csv(
        tmp_path / "seed_101" / "validation_runs.csv",
        [{"policy": "PPO", "seed": 10000, "weighted_tardiness": 80.0}],
    )
    with pytest.raises(ValueError, match="incomplete or leaked"):
        validate_member(cfg, tmp_path, 101)


def test_representative_member_is_closest_to_median_not_best() -> None:
    rows = [
        {
            "training_seed": 101,
            "weighted_tardiness_mean": 60.0,
            "mean_decision_time_ms_mean": 1.0,
        },
        {
            "training_seed": 202,
            "weighted_tardiness_mean": 80.0,
            "mean_decision_time_ms_mean": 1.0,
        },
        {
            "training_seed": 303,
            "weighted_tardiness_mean": 120.0,
            "mean_decision_time_ms_mean": 1.0,
        },
    ]
    selected = select_representative_member(rows)
    assert selected["training_seed"] == 202


def test_aggregate_reports_training_seed_dispersion() -> None:
    rows = [
        {"weighted_tardiness_mean": 60.0, "mean_decision_time_ms_mean": 1.0},
        {"weighted_tardiness_mean": 80.0, "mean_decision_time_ms_mean": 2.0},
        {"weighted_tardiness_mean": 100.0, "mean_decision_time_ms_mean": 3.0},
    ]
    result = aggregate_member_metrics(rows)
    assert result["weighted_tardiness_training_mean"] == pytest.approx(80.0)
    assert result["weighted_tardiness_training_median"] == pytest.approx(80.0)
    assert result["weighted_tardiness_training_std"] > 0.0
