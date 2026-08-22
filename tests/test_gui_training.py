import json

import pytest

from so_snake.gui.training import TrainingManager


def test_model_library_reads_policy_and_never_reuses_a_name(tmp_path):
    manager = TrainingManager(tmp_path / "outputs")
    checkpoint = tmp_path / "outputs" / "act_run-20260822-120000" / "checkpoints" / "last" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(json.dumps({"type": "act"}))
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint.parents[2] / "so_snake_observation.json").write_text(
        json.dumps({"version": 1, "roi": {"wrist": [0.1, 0.2, 0.7, 0.6]}})
    )

    models = manager.models()["models"]
    assert models[0]["policy"] == "act"
    assert models[0]["ready"] is True
    assert models[0]["roi"] == {"wrist": [0.1, 0.2, 0.7, 0.6]}
    assert manager._unique_name("act_run") != "act_run-20260822-120000"


def test_pi05_rejects_mps_before_starting_a_job(tmp_path):
    manager = TrainingManager(tmp_path / "outputs")
    with pytest.raises(ValueError, match="not supported on MPS"):
        manager.start_train(
            dataset=tmp_path / "missing", policy="pi05", name="pi", device="mps",
            steps=1, batch_size=1, base_model="lerobot/pi05_base",
        )


def test_wandb_args_are_opt_in_and_validated():
    assert TrainingManager._wandb_args({"enabled": False}) == []
    assert TrainingManager._wandb_args({"enabled": True, "project": "so-snake", "entity": "lab"}) == [
        "--wandb.enable=true", "--wandb.project=so-snake", "--wandb.entity=lab"
    ]
    with pytest.raises(ValueError, match="project"):
        TrainingManager._wandb_args({"enabled": True})


def test_training_observation_profile_carries_export_roi(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "export.json").write_text(
        json.dumps({"roi": {"third_person": [0.0, 0.1, 1.0, 0.8]}, "resolution": [240, 320]})
    )
    assert TrainingManager._observation_profile(dataset)["roi"] == {
        "third_person": [0.0, 0.1, 1.0, 0.8]
    }


def test_rollout_receives_the_model_roi(tmp_path, monkeypatch):
    manager = TrainingManager(tmp_path / "outputs")
    checkpoint = tmp_path / "outputs" / "run" / "checkpoints" / "last" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint.parents[2] / "so_snake_observation.json").write_text(
        json.dumps({"roi": {"wrist": [0.1, 0.2, 0.7, 0.6]}})
    )
    captured = {}
    monkeypatch.setattr(manager, "_start", lambda **kwargs: captured.update(kwargs) or kwargs)

    manager.start_rollout(checkpoint=checkpoint, task="pick", argv_tail=[])

    assert "--roi" in captured["argv"]
    assert "wrist=0.1,0.2,0.7,0.6" in captured["argv"]
