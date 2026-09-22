from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from dmdtrl import final_execute


def test_load_named_ppo_policy_sets_analysis_name(monkeypatch) -> None:
    dummy = SimpleNamespace(policy_name="PPO")

    def fake_loader(path: Path):
        assert path == Path("model.zip")
        return dummy

    monkeypatch.setattr("dmdtrl.policies.load_ppo_policy", fake_loader)
    loaded = final_execute.load_named_ppo_policy(Path("model.zip"), 303)
    assert loaded is dummy
    assert loaded.policy_name == "PPO_TRAIN_303"
