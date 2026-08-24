from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.hub.push import (
    PUSH_FILENAMES,
    Checkpoint,
    discover,
    render_collection_card,
    render_model_card,
)


@pytest.fixture
def fixture_root(tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs"
    checkpoints_dir = tmp_path / "checkpoints"
    config_dir.mkdir()
    (config_dir / "qwen3_5_2b.yaml").write_text(
        yaml.dump(
            {
                "model": {"id": "Qwen/Qwen3.5-2B"},
                "lora": {
                    "r": 8,
                    "alpha": 16,
                    "dropout": 0.05,
                    "target_modules": ["q_proj", "v_proj"],
                },
            }
        )
    )

    done_dir = checkpoints_dir / "english" / "qwen3_5_2b" / "cultural"
    done_dir.mkdir(parents=True)
    (done_dir / "TRAINING_DONE").touch()
    (done_dir / "adapter_model.safetensors").write_bytes(b"fake")
    for extra in ("training_args.bin", "mlflow_run_id.txt"):
        (done_dir / extra).touch()
    (done_dir / "checkpoint-100").mkdir()
    (done_dir / "checkpoint-100" / "adapter_model.safetensors").write_bytes(b"fake")

    incomplete_dir = checkpoints_dir / "german" / "qwen3_5_2b" / "cultural"
    incomplete_dir.mkdir(parents=True)
    (incomplete_dir / "adapter_model.safetensors").write_bytes(b"fake")

    return tmp_path


def _discover(root: Path) -> list[Checkpoint]:
    return discover(
        checkpoints_dir=root / "checkpoints",
        cultures=("english", "german"),
        backbones=("qwen3_5_2b",),
        problem="cultural",
        config_dir=root / "configs",
    )


def test_discover_only_returns_checkpoints_with_the_done_sentinel(fixture_root: Path) -> None:
    found = _discover(fixture_root)

    assert len(found) == 1
    assert found[0].culture == "english"
    assert found[0].backbone == "qwen3_5_2b"
    assert found[0].base_model_id == "Qwen/Qwen3.5-2B"
    assert found[0].path_in_repo == "english/qwen3_5_2b/cultural"


def test_push_filenames_exclude_training_artifacts() -> None:
    assert "adapter_model.safetensors" in PUSH_FILENAMES
    assert "training_args.bin" not in PUSH_FILENAMES
    assert "mlflow_run_id.txt" not in PUSH_FILENAMES
    assert "TRAINING_DONE" not in PUSH_FILENAMES


def test_model_card_names_the_base_model_and_lora_config(fixture_root: Path) -> None:
    ckpt = _discover(fixture_root)[0]
    card = render_model_card(ckpt)

    assert "base_model: Qwen/Qwen3.5-2B" in card
    assert "culture:english" in card
    assert "r: 8" in card
    assert 'subfolder="english/qwen3_5_2b/cultural"' in card


def test_collection_card_marks_only_the_pushed_pairs(fixture_root: Path) -> None:
    found = _discover(fixture_root)
    card = render_collection_card(found, "Neemias/Culture-Steering-MLLM-Collection")

    assert "| english | ✅ |" in card
    assert "german" not in card
