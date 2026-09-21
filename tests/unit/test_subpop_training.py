"""Pin the SubPOP training splits, its metrics and the shipped recipe."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import yaml
from transformers import EvalPrediction

from src.data.subpop_data import (
    DATASET_FILES,
    DATASET_REVISION,
    SOURCE_COMMIT,
    STEERING_FILENAME,
    SubpopEncoder,
    load_steering,
)
from src.training import train_subpop
from src.training.train_subpop import LOSS_NAME, compute_metrics, load_training_splits, train

CONFIG_DIR = Path("configs")
SUBPOP_CONFIGS = sorted(CONFIG_DIR.glob("*_subpop.yaml"))
RecordFactory = Callable[..., dict[str, Any]]
TokenizerFactory = Callable[..., Any]


def _write_tables(
    raw_dir: Path, raw_subpop_record: RecordFactory, steering: list[dict[str, str]]
) -> None:
    raw_dir.mkdir()
    (raw_dir / STEERING_FILENAME).write_text(json.dumps(steering), encoding="utf-8")
    rows = [
        raw_subpop_record(qkey=f"Q{index}_W1", group=group)
        for index in range(10)
        for group in ("Liberal", "Moderate")
    ]
    rows += [
        raw_subpop_record(
            qkey="ATTENDPERSON2_W117",
            group=group,
            question="In general, how often do you attend religious services in person?",
        )
        for group in ("Liberal", "Moderate")
    ]
    (raw_dir / DATASET_FILES["train"]).write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def _encoder(raw_dir: Path, survey_tokenizer: TokenizerFactory) -> SubpopEncoder:
    return SubpopEncoder.build(survey_tokenizer(), load_steering(raw_dir), max_length=256)


def test_load_training_splits_follows_the_protocol_unless_asked_to_filter(
    tmp_path: Path,
    survey_tokenizer: TokenizerFactory,
    raw_subpop_record: RecordFactory,
    subpop_steering_entries: list[dict[str, str]],
) -> None:
    raw_dir = tmp_path / "raw"
    _write_tables(raw_dir, raw_subpop_record, subpop_steering_entries)
    encoder = _encoder(raw_dir, survey_tokenizer)

    train_set, valid_set, stats = load_training_splits(
        encoder, raw_dir=raw_dir, exclude_related=False, debug=False
    )
    filtered_train, filtered_valid, filtered_stats = load_training_splits(
        encoder, raw_dir=raw_dir, exclude_related=True, debug=False
    )

    assert train_set.num_rows + valid_set.num_rows == 22
    assert (stats["train_questions"], stats["val_questions"]) == (10, 1)
    assert valid_set.num_rows == 2
    assert stats["dropped_rows"] == 0
    assert stats["val_ratio"] == pytest.approx(0.1)
    assert stats["split_seed"] == 42
    assert stats["dataset_revision"] == DATASET_REVISION
    assert stats["source_commit"] == SOURCE_COMMIT
    assert filtered_train.num_rows + filtered_valid.num_rows == 20
    assert filtered_stats["dropped_rows"] == 2
    assert filtered_stats["related_questions_in_train"] == (
        "happiness=0,social trust=0,religious attendance=0,left-right self-placement=0"
    )


def test_debug_keeps_only_a_small_slice_of_each_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    survey_tokenizer: TokenizerFactory,
    raw_subpop_record: RecordFactory,
    subpop_steering_entries: list[dict[str, str]],
) -> None:
    raw_dir = tmp_path / "raw"
    _write_tables(raw_dir, raw_subpop_record, subpop_steering_entries)
    monkeypatch.setattr(train_subpop, "DEBUG_TRAIN_ROWS", 3)
    monkeypatch.setattr(train_subpop, "DEBUG_VALID_ROWS", 1)

    train_set, valid_set, stats = load_training_splits(
        _encoder(raw_dir, survey_tokenizer), raw_dir=raw_dir, exclude_related=False, debug=True
    )

    assert (train_set.num_rows, valid_set.num_rows) == (3, 1)
    assert stats["train_rows"] == 3


def test_compute_metrics_reports_kl_entropy_and_mass_without_an_ordinal_distance() -> None:
    logits = np.zeros((2, 10))
    target = np.zeros((2, 10))
    target[0, :2] = [0.5, 0.5]
    target[1, 0] = 1.0

    mass = np.array([0.4, 0.6])

    metrics = compute_metrics(
        EvalPrediction(predictions=cast(Any, (logits, mass)), label_ids=target)
    )

    assert set(metrics) == {"kl", "entropy", "option_mass"}
    assert metrics["kl"] == pytest.approx((math.log(5.0) + math.log(10.0)) / 2)
    assert metrics["entropy"] == pytest.approx(math.log(10.0))
    assert metrics["option_mass"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="option_logits, option_mass"):
        compute_metrics(EvalPrediction(predictions=logits, label_ids=target))


def test_the_trainer_refuses_a_config_written_for_another_track() -> None:
    cfg = {"model": {}, "lora": {}, "training": {}, "data": {"format": "distributional"}}

    with pytest.raises(ValueError, match=r"data\.format must be 'subpop'"):
        train(cfg, "qwen3_vl_2b", debug=False)


def test_the_logged_loss_name_says_what_is_normalised() -> None:
    assert LOSS_NAME == "first_token_forward_kl_over_ten_letters"


def _load(path: Path) -> dict[str, Any]:
    with open(path) as handle:
        return dict(yaml.safe_load(handle))


@pytest.fixture(params=SUBPOP_CONFIGS, ids=lambda p: p.stem)
def subpop_config(request: pytest.FixtureRequest) -> tuple[Path, dict[str, Any]]:
    path = Path(request.param)
    return path, _load(path)


def test_the_deliverable_backbone_has_a_subpop_config() -> None:
    assert CONFIG_DIR / "qwen3_vl_2b_subpop.yaml" in SUBPOP_CONFIGS


def test_a_subpop_config_names_its_track_and_pins_the_backbone_key(
    subpop_config: tuple[Path, dict[str, Any]],
) -> None:
    path, cfg = subpop_config

    assert cfg["condition"] == "subpop"
    assert cfg["data"]["format"] == "subpop"
    assert cfg["data"]["val_ratio"] == pytest.approx(0.1)
    assert cfg["data"]["split_seed"] == 42
    assert isinstance(cfg["data"]["exclude_related_items"], bool)
    assert cfg["model"]["key"] == path.stem.removesuffix("_subpop")
    assert "cultures" not in cfg
    assert "n_folds" not in cfg
    assert cfg["mlflow"]["experiment"] == "culture_mllm_subpop_training"


def test_a_subpop_config_loads_the_backbone_exactly_as_the_wvs_config_does(
    subpop_config: tuple[Path, dict[str, Any]],
) -> None:
    _, cfg = subpop_config
    base = _load(CONFIG_DIR / f"{cfg['model']['key']}.yaml")

    assert {key: value for key, value in cfg["model"].items() if key != "key"} == base["model"]


def test_a_subpop_config_follows_the_reference_recipe(
    subpop_config: tuple[Path, dict[str, Any]],
) -> None:
    _, cfg = subpop_config
    train_cfg = cfg["training"]

    assert cfg["lora"] == {
        "r": 8,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
    }
    assert train_cfg["learning_rate"] == pytest.approx(2e-4)
    assert train_cfg["max_epochs"] == 50
    assert train_cfg["batch_size"] * train_cfg["gradient_accumulation"] == 128
    assert train_cfg["lr_schedule"] == "cosine"
    assert "lr_gamma" not in train_cfg
    assert train_cfg["warmup_steps"] == pytest.approx(0.1)
    assert train_cfg["max_seq_len"] >= 312
    assert train_cfg["save_steps"] % train_cfg["eval_steps"] == 0
