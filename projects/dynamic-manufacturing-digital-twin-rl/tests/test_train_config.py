import json
from pathlib import Path

import pytest

from dmdtrl.env import EnvConfig
from dmdtrl.train import (
    PPOTrainingConfig,
    build_training_manifest,
    default_manifest_path,
    model_archive_path,
    write_manifest,
)


def test_training_config_validates_and_builds_ppo_kwargs():
    config = PPOTrainingConfig(total_timesteps=2_048, n_steps=256, batch_size=64, verbose=0)
    kwargs = config.ppo_kwargs()
    assert kwargs["n_steps"] == 256
    assert kwargs["batch_size"] == 64
    assert kwargs["policy_kwargs"] == {"net_arch": [128, 128]}
    assert kwargs["device"] == "cpu"


@pytest.mark.parametrize(
    "config",
    [
        PPOTrainingConfig(total_timesteps=0),
        PPOTrainingConfig(seed=-1),
        PPOTrainingConfig(learning_rate=0.0),
        PPOTrainingConfig(n_steps=1),
        PPOTrainingConfig(n_steps=32, batch_size=64),
        PPOTrainingConfig(gamma=0.0),
        PPOTrainingConfig(gae_lambda=1.1),
        PPOTrainingConfig(ent_coef=-0.1),
        PPOTrainingConfig(hidden_units=0),
    ],
)
def test_training_config_rejects_invalid_values(config):
    with pytest.raises(ValueError):
        config.validate()


def test_model_and_manifest_paths_are_predictable():
    assert model_archive_path(Path("models/policy")) == Path("models/policy.zip")
    assert model_archive_path(Path("models/policy.zip")) == Path("models/policy.zip")
    assert default_manifest_path(Path("models/policy")) == Path("models/policy_manifest.json")


def test_training_manifest_records_reproducibility_contract(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/test")
    training = PPOTrainingConfig(total_timesteps=1_024, seed=7, n_steps=256, batch_size=64)
    env_config = EnvConfig(n_jobs=12, n_machines=3)

    manifest = build_training_manifest(
        training,
        env_config,
        model_path=Path("models/policy.zip"),
        training_seconds=1.25,
    )

    assert manifest["algorithm"] == "PPO"
    assert manifest["training_regime"] == "nominal_environment"
    assert manifest["training_seed"] == 7
    assert manifest["training_config"]["total_timesteps"] == 1_024
    assert manifest["environment_config"]["n_jobs"] == 12
    assert manifest["git"]["sha"] == "abc123"
    assert manifest["evaluation_seed_convention"]["nominal_test_seed_start"] == 20_000
    assert manifest["evaluation_seed_convention"]["stress_test_seed_start"] == 30_000


def test_write_manifest_creates_json_file(tmp_path):
    output = tmp_path / "nested" / "manifest.json"
    write_manifest({"answer": 42}, output)
    assert json.loads(output.read_text(encoding="utf-8")) == {"answer": 42}
