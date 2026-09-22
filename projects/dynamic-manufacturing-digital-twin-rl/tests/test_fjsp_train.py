from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from dmdtrl.fjsp_env import FJSPEnvConfig, FlexibleJobShopEnv
from dmdtrl.fjsp_generator import FJSPGeneratorConfig
from dmdtrl.fjsp_train import (
    FJSPMaskablePPOConfig,
    build_parser,
    build_training_manifest,
    default_manifest_path,
    model_archive_path,
    train_maskable_ppo,
    validate_mask_contract,
)


def _env_config() -> FJSPEnvConfig:
    return FJSPEnvConfig(
        generator=FJSPGeneratorConfig(
            n_jobs=4,
            n_machines=3,
            n_families=2,
            operations_min=2,
            operations_max=3,
            eligible_machines_min=1,
            eligible_machines_max=2,
            processing_min=1.0,
            processing_max=3.0,
        ),
        default_setup_time=0.5,
    )


def test_maskable_ppo_config_validation_and_kwargs() -> None:
    config = FJSPMaskablePPOConfig(total_timesteps=1024, n_steps=256, batch_size=64)
    kwargs = config.algorithm_kwargs()
    assert kwargs["n_steps"] == 256
    assert kwargs["batch_size"] == 64
    assert kwargs["policy_kwargs"]["net_arch"]["pi"] == [128, 128]

    invalid = [
        FJSPMaskablePPOConfig(total_timesteps=0),
        FJSPMaskablePPOConfig(seed=-1),
        FJSPMaskablePPOConfig(learning_rate=0.0),
        FJSPMaskablePPOConfig(n_steps=1),
        FJSPMaskablePPOConfig(n_steps=64, batch_size=128),
        FJSPMaskablePPOConfig(n_steps=250, batch_size=64),
        FJSPMaskablePPOConfig(gamma=0.0),
        FJSPMaskablePPOConfig(gae_lambda=1.1),
        FJSPMaskablePPOConfig(ent_coef=-0.1),
        FJSPMaskablePPOConfig(hidden_units=0),
    ]
    for item in invalid:
        with pytest.raises(ValueError):
            item.validate()


def test_model_and_manifest_paths() -> None:
    assert model_archive_path("models/fjsp") == Path("models/fjsp.zip")
    assert model_archive_path("models/fjsp.zip") == Path("models/fjsp.zip")
    assert default_manifest_path("models/fjsp") == Path("models/fjsp_manifest.json")


def test_mask_contract_completes_generated_episode() -> None:
    env = FlexibleJobShopEnv(_env_config())
    result = validate_mask_contract(env, seed=601)
    assert result["validated_decisions"] >= 8
    assert result["min_feasible_actions"] >= 1
    assert result["max_feasible_actions"] >= result["min_feasible_actions"]


def test_training_manifest_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    config = FJSPMaskablePPOConfig(total_timesteps=1024, n_steps=256, batch_size=64, seed=777)
    manifest = build_training_manifest(
        training_config=config,
        env_config=_env_config(),
        model_path=Path("model.zip"),
        training_seconds=1.25,
        mask_contract={
            "validated_decisions": 10,
            "min_feasible_actions": 1,
            "max_feasible_actions": 4,
        },
    )
    assert manifest["algorithm"] == "MaskablePPO"
    assert manifest["training_seed"] == 777
    assert manifest["environment_config"]["generator"]["n_jobs"] == 4
    assert manifest["phase5_seed_policy"]["validation"] == "41000-41999"
    assert manifest["git"]["sha"] == "abc123"


def test_train_maskable_ppo_with_fake_external_algorithm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeMaskablePPO:
        def __init__(self, policy, env, **kwargs):
            assert policy == "MlpPolicy"
            assert kwargs["seed"] == 601
            self.env = env
            self.learn_steps = None

        def learn(self, total_timesteps):
            self.learn_steps = total_timesteps
            return self

        def save(self, path):
            Path(f"{path}.zip").write_bytes(b"fake-model")

    fake_module = types.ModuleType("sb3_contrib")
    fake_module.MaskablePPO = FakeMaskablePPO
    monkeypatch.setitem(sys.modules, "sb3_contrib", fake_module)

    output = tmp_path / "maskable"
    metadata = tmp_path / "manifest.json"
    archive, manifest_path = train_maskable_ppo(
        output,
        training_config=FJSPMaskablePPOConfig(
            total_timesteps=512,
            n_steps=128,
            batch_size=64,
            seed=601,
            verbose=0,
        ),
        env_config=_env_config(),
        metadata_path=metadata,
    )
    assert archive.exists()
    assert manifest_path == metadata
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert payload["training_config"]["total_timesteps"] == 512
    assert payload["mask_contract"]["validated_decisions"] > 0


def test_cli_parser_accepts_small_smoke_configuration() -> None:
    args = build_parser().parse_args(
        [
            "--steps",
            "2048",
            "--seed",
            "601",
            "--jobs",
            "6",
            "--machines",
            "3",
            "--operations-max",
            "3",
            "--eligible-max",
            "2",
            "--quiet",
        ]
    )
    assert args.steps == 2048
    assert args.jobs == 6
    assert args.quiet is True
