from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from dmdtrl.fjsp_env import FJSPEnvConfig
from dmdtrl.fjsp_generator import FJSPGeneratorConfig
from dmdtrl.fjsp_hh_train import (
    FJSPHyperHeuristicPPOConfig,
    build_training_manifest,
    model_archive_path,
    train_hyperheuristic_ppo,
    validate_operator_contract,
)
from dmdtrl.fjsp_hyperheuristic_env import FlexibleJobShopHyperHeuristicEnv
from dmdtrl.fjsp_operators import FJSPOperator


def _env_config() -> FJSPEnvConfig:
    return FJSPEnvConfig(
        generator=FJSPGeneratorConfig(
            n_jobs=5,
            n_machines=3,
            n_families=3,
            operations_min=2,
            operations_max=3,
            eligible_machines_min=1,
            eligible_machines_max=2,
        ),
        default_setup_time=0.5,
    )


def test_training_config_validation_and_archive_path() -> None:
    config = FJSPHyperHeuristicPPOConfig(total_timesteps=1_000, n_steps=256, batch_size=64)
    config.validate()
    assert model_archive_path("model") == Path("model.zip")
    assert model_archive_path("model.zip") == Path("model.zip")
    kwargs = config.algorithm_kwargs()
    assert kwargs["n_steps"] == 256
    assert kwargs["batch_size"] == 64
    assert kwargs["policy_kwargs"]["net_arch"]["pi"] == [128, 128]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"total_timesteps": 0}, "total_timesteps"),
        ({"seed": -1}, "seed"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"n_steps": 1}, "n_steps"),
        ({"n_steps": 64, "batch_size": 128}, "batch_size"),
        ({"n_steps": 250, "batch_size": 64}, "divisible"),
        ({"gamma": 0.0}, "gamma"),
        ({"gae_lambda": 1.5}, "gae_lambda"),
        ({"ent_coef": -0.1}, "ent_coef"),
        ({"hidden_units": 0}, "hidden_units"),
    ],
)
def test_training_config_rejects_invalid_values(
    overrides: dict[str, int | float],
    message: str,
) -> None:
    values = {
        "total_timesteps": 1_000,
        "seed": 801,
        "learning_rate": 3e-4,
        "n_steps": 256,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "hidden_units": 128,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        FJSPHyperHeuristicPPOConfig(**values).validate()


def test_operator_contract_completes_episode() -> None:
    env = FlexibleJobShopHyperHeuristicEnv(_env_config())
    contract = validate_operator_contract(env, seed=40030)
    assert contract["operator_action_count"] == len(FJSPOperator)
    assert contract["validated_decisions"] > 0


def test_training_manifest_declares_action_abstraction_and_seed_embargo() -> None:
    training = FJSPHyperHeuristicPPOConfig(
        total_timesteps=4_096,
        seed=801,
        n_steps=256,
        batch_size=64,
    )
    manifest = build_training_manifest(
        training_config=training,
        env_config=_env_config(),
        model_path=Path("artifacts/fjsp_hh_ppo.zip"),
        training_seconds=1.25,
        operator_contract={
            "validated_decisions": 12,
            "operator_action_count": len(FJSPOperator),
        },
    )

    assert manifest["algorithm"] == "PPO"
    assert manifest["controller"] == "FJSP_HYPER_HEURISTIC"
    assert manifest["operator_contract"]["operator_action_count"] == len(FJSPOperator)
    assert len(manifest["operator_names"]) == len(FJSPOperator)
    assert manifest["phase5_seed_policy"]["development_evaluation"] == "40000-40999"
    assert "embargoed" in manifest["phase5_seed_policy"]["final"]


def test_training_function_with_fake_ppo_persists_model_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = types.ModuleType("stable_baselines3")

    class FakePPO:
        def __init__(self, policy, env, **kwargs):  # noqa: ANN001, ANN003, ANN204
            assert policy == "MlpPolicy"
            assert env.action_space.n == len(FJSPOperator)
            self.kwargs = kwargs

        def learn(self, total_timesteps: int):  # noqa: ANN201
            assert total_timesteps == 1_024
            return self

        def save(self, path: str) -> None:
            Path(f"{path}.zip").write_bytes(b"fake-ppo-model")

    fake_module.PPO = FakePPO
    monkeypatch.setitem(sys.modules, "stable_baselines3", fake_module)

    training = FJSPHyperHeuristicPPOConfig(
        total_timesteps=1_024,
        seed=801,
        n_steps=256,
        batch_size=64,
        verbose=0,
    )
    archive, manifest_path = train_hyperheuristic_ppo(
        tmp_path / "model",
        training_config=training,
        env_config=_env_config(),
        metadata_path=tmp_path / "manifest.json",
    )

    assert archive.read_bytes() == b"fake-ppo-model"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["algorithm"] == "PPO"
    assert manifest["training_seed"] == 801
    assert manifest["operator_contract"]["operator_action_count"] == len(FJSPOperator)
