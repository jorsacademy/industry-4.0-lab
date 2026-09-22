from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from dmdtrl import fjsp_hh_campaign, fjsp_hh_member
from dmdtrl.fjsp_hh_protocol import load_hh_validation_design
from dmdtrl.fjsp_operators import OPERATOR_NAMES
from dmdtrl.fjsp_ppo_campaign import write_csv

DESIGN_PATH = Path("configs/fjsp_hh_validation_design.json")
FREEZE_PATH = Path("configs/fjsp_cpsat_validation_freeze.json")


def _design() -> dict:
    return load_hh_validation_design(DESIGN_PATH)


def _freeze() -> dict:
    return json.loads(FREEZE_PATH.read_text(encoding="utf-8"))


def test_builders_apply_complete_predeclared_configuration() -> None:
    design = _design()
    generator = fjsp_hh_campaign.build_generator_config(design)
    env = fjsp_hh_campaign.build_env_config(design)
    training = fjsp_hh_campaign.build_training_config(design, 901)

    assert asdict(generator) == design["environment_config"]["generator"]
    assert asdict(env) == design["environment_config"]
    for key, value in design["training_config"].items():
        assert getattr(training, key) == value
    assert training.seed == 901
    assert training.verbose == 0

    with pytest.raises(ValueError, match="predeclared"):
        fjsp_hh_campaign.build_training_config(design, 902)


def test_validation_seeds_and_baseline_policy_set_are_frozen() -> None:
    design = _design()
    assert fjsp_hh_campaign.validation_seeds(design) == list(range(41_200, 41_230))
    assert fjsp_hh_campaign.baseline_policies(design) == (
        *OPERATOR_NAMES,
        "ROLLING_HORIZON_CPSAT",
    )


def test_frozen_cpsat_configuration_is_reused_without_retuning() -> None:
    design = _design()
    freeze = _freeze()
    config = fjsp_hh_campaign.validate_cpsat_freeze(freeze, design)
    assert config.job_horizon == 4
    assert config.solver_seconds == 0.10
    assert config.num_search_workers == 1

    changed = dict(freeze)
    changed["selected_policy"] = "OTHER"
    with pytest.raises(ValueError, match="does not match"):
        fjsp_hh_campaign.validate_cpsat_freeze(changed, design)


def test_baseline_panel_uses_all_operators_and_common_instances(monkeypatch) -> None:
    design = _design()

    def fake_operator(instance, operator, *, setup_time):
        assert setup_time == 1.0
        return {
            "policy": operator.name,
            "weighted_tardiness": float(operator.value),
            "makespan": 1.0,
            "mean_flow_time": 1.0,
            "mean_decision_time_ms": 0.1,
        }

    def fake_cpsat(instance, *, setup_time, config):
        assert setup_time == 1.0
        assert config.job_horizon == 4
        return {
            "policy": "ROLLING_HORIZON_CPSAT",
            "weighted_tardiness": 0.0,
            "makespan": 1.0,
            "mean_flow_time": 1.0,
            "mean_decision_time_ms": 1.0,
        }

    monkeypatch.setattr(fjsp_hh_campaign, "_run_operator", fake_operator)
    monkeypatch.setattr(fjsp_hh_campaign, "_run_cpsat", fake_cpsat)
    rows = fjsp_hh_campaign.evaluate_baseline_panel(design, _freeze())
    assert len(rows) == 30 * 9
    assert {row["policy"] for row in rows} == set(fjsp_hh_campaign.baseline_policies(design))
    for seed in range(41_200, 41_230):
        fingerprints = {
            row["instance_sha256"] for row in rows if int(row["seed"]) == seed
        }
        assert len(fingerprints) == 1


def _baseline_artifacts(root: Path, design: dict) -> dict[int, str]:
    policies = fjsp_hh_campaign.baseline_policies(design)
    fingerprints = {seed: f"fp-{seed}" for seed in fjsp_hh_campaign.validation_seeds(design)}
    rows = []
    for seed, fingerprint in fingerprints.items():
        for index, policy in enumerate(policies):
            rows.append(
                {
                    "seed": seed,
                    "seed_regime": "validation",
                    "instance_sha256": fingerprint,
                    "policy": policy,
                    "weighted_tardiness": float(10 + index),
                    "makespan": 20.0,
                    "mean_flow_time": 5.0,
                    "mean_decision_time_ms": 0.5,
                }
            )
    summary = [{"policy": policy, "weighted_tardiness_mean": 10.0} for policy in policies]
    root.mkdir(parents=True)
    write_csv(rows, root / "baseline_runs.csv")
    write_csv(summary, root / "baseline_summary.csv")
    return fingerprints


def _member_artifacts(
    root: Path,
    design: dict,
    training_seed: int,
    fingerprints: dict[int, str],
    *,
    wtt: float,
) -> None:
    member = root / f"seed_{training_seed}"
    member.mkdir(parents=True)
    (member / "fjsp_hyperheuristic_ppo.zip").write_bytes(b"real-model-placeholder")
    manifest = {
        "algorithm": "PPO",
        "controller": "FJSP_HYPER_HEURISTIC",
        "training_seed": training_seed,
        "training_config": {**design["training_config"], "seed": training_seed, "verbose": 0},
        "environment_config": design["environment_config"],
        "training_seconds": 12.0,
        "runtime": {"stable_baselines3": "2.9.0"},
    }
    (member / "training_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    rows = [
        {
            "training_seed": training_seed,
            "seed": seed,
            "seed_regime": "validation",
            "instance_sha256": fingerprints[seed],
            "policy": "PPO_HYPER_HEURISTIC",
            "weighted_tardiness": wtt + (seed % 3),
            "makespan": 20.0,
            "mean_flow_time": 5.0,
            "mean_decision_time_ms": 0.3,
        }
        for seed in fjsp_hh_campaign.validation_seeds(design)
    ]
    write_csv(rows, member / "ppo_validation_runs.csv")
    write_csv(
        [
            {
                "training_seed": training_seed,
                "policy": "PPO_HYPER_HEURISTIC",
                "weighted_tardiness_mean": wtt,
                "weighted_tardiness_std": 1.0,
                "weighted_tardiness_ci_low": wtt - 1.0,
                "weighted_tardiness_ci_high": wtt + 1.0,
                "makespan_mean": 20.0,
                "mean_flow_time": 5.0,
                "mean_decision_time_ms": 0.3,
            }
        ],
        member / "ppo_validation_summary.csv",
    )


def test_member_validation_enforces_common_instances_and_manifest(tmp_path: Path) -> None:
    design = _design()
    baseline = tmp_path / "baseline"
    members = tmp_path / "members"
    fingerprints = _baseline_artifacts(baseline, design)
    _member_artifacts(members, design, 901, fingerprints, wtt=7.0)

    _, _, observed = fjsp_hh_campaign.validate_baseline_panel(design, baseline)
    member, rows = fjsp_hh_campaign.validate_member(design, members, 901, observed)
    assert member["training_seed"] == 901
    assert member["n_validation_seeds"] == 30
    assert len(rows) == 30

    bad_path = members / "seed_901" / "ppo_validation_runs.csv"
    bad_rows = fjsp_hh_campaign.read_csv(bad_path)
    bad_rows[0]["instance_sha256"] = "wrong"
    write_csv(bad_rows, bad_path)
    with pytest.raises(ValueError, match="different common-seed"):
        fjsp_hh_campaign.validate_member(design, members, 901, observed)


def test_training_seed_aggregation_never_selects_best_seed_as_result(tmp_path: Path) -> None:
    design = _design()
    baseline = tmp_path / "baseline"
    members = tmp_path / "members"
    fingerprints = _baseline_artifacts(baseline, design)
    for index, seed in enumerate(design["training_seeds"]):
        _member_artifacts(members, design, seed, fingerprints, wtt=5.0 + index)

    member_rows, runs, comparisons, manifest = fjsp_hh_campaign.run_aggregate(
        design=design,
        freeze=_freeze(),
        baseline_root=baseline,
        members_root=members,
        n_bootstrap=100,
        n_permutations=100,
    )
    assert len(member_rows) == 5
    assert len(runs) == 30 * (9 + 5)
    assert len(comparisons) == 5 * 9
    aggregate = manifest["aggregate_training_seed_comparisons"]
    assert len(aggregate) == 9
    assert all(row["n_training_seeds"] == 5 for row in aggregate)
    assert all(row["inference_unit"] == "training_seed" for row in aggregate)
    assert manifest["representative_training_seed"] == 2901
    assert manifest["representative_model_role"] == "deployment_and_demo_only"
    assert manifest["final_test_used_for_selection"] is False


def test_member_evaluator_is_validation_only_and_common_seed(monkeypatch) -> None:
    design = _design()

    def fake_run(model, *, seed, expected_fingerprint, env_config):
        assert model == "model"
        assert seed in range(41_200, 41_230)
        assert expected_fingerprint
        assert env_config.generator.n_jobs == 12
        return {
            "policy": "PPO_HYPER_HEURISTIC",
            "weighted_tardiness": 1.0,
            "makespan": 2.0,
            "mean_flow_time": 1.0,
            "mean_decision_time_ms": 0.2,
        }

    monkeypatch.setattr(fjsp_hh_member, "_run_ppo", fake_run)
    rows = fjsp_hh_member.evaluate_member("model", design=design, training_seed=901)
    assert [int(row["seed"]) for row in rows] == list(range(41_200, 41_230))
    assert all(row["seed_regime"] == "validation" for row in rows)

    with pytest.raises(ValueError, match="predeclared"):
        fjsp_hh_member.evaluate_member("model", design=design, training_seed=902)
