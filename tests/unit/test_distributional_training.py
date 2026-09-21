"""Pin the first-token KL objective, its metrics and the trainer hooks around them."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
import yaml
from transformers import EvalPrediction

from src.data.distributional_data import (
    MAX_OPTIONS,
    SPLIT_FILES,
    TARGET_PAD,
    DistributionalCollator,
    PromptEncoder,
)
from src.training import train_distributional
from src.training.train_distributional import (
    DistributionalTrainer,
    answer_positions,
    assert_base_elicits_options,
    compute_metrics,
    exact_kl,
    forward_answer_logits,
    kl_loss,
    load_training_splits,
    option_entropy,
    option_mass,
    ordinal_emd,
    predicted_distribution,
    score_batch,
    step_epoch_factor,
    uniform_kl,
)

CONFIG_DIR = Path("configs")
DISTRIBUTIONAL_CONFIGS = sorted(CONFIG_DIR.glob("*_distributional.yaml"))
RecordFactory = Callable[..., dict[str, Any]]
TokenizerFactory = Callable[..., Any]


def _padded(rows: list[list[float]], fill: float = TARGET_PAD) -> torch.Tensor:
    out = torch.full((len(rows), MAX_OPTIONS), fill)
    for index, row in enumerate(rows):
        out[index, : len(row)] = torch.tensor(row)
    return out


def _kl(p: list[float], q: list[float]) -> float:
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q, strict=True) if pi > 0)


KL_A = _kl([0.25, 0.75], [0.5, 0.5])
KL_B = _kl([1.0, 0.0, 0.0], [0.5, 0.25, 0.25])


def _two_record_case() -> tuple[torch.Tensor, torch.Tensor]:
    logits = _padded([[0.0, 0.0], [math.log(2.0), 0.0, 0.0]], fill=100.0)
    target = _padded([[0.25, 0.75], [1.0, 0.0, 0.0]])
    return logits, target


def test_kl_loss_matches_the_reference_normalisation_by_hand() -> None:
    logits, target = _two_record_case()

    assert kl_loss(logits, target).item() == pytest.approx((KL_A / 2 + KL_B / 3) / 2)
    assert exact_kl(logits, target).item() == pytest.approx((KL_A + KL_B) / 2)


def test_kl_loss_is_zero_when_the_prediction_matches_a_target_with_a_zero_option() -> None:
    target = _padded([[0.5, 0.5, 0.0]])
    logits = _padded([[0.0, 0.0, -1e4]], fill=100.0)

    loss = kl_loss(logits, target)

    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert torch.isfinite(loss)


def test_kl_loss_ignores_whatever_sits_in_padded_slots() -> None:
    target = _padded([[0.3, 0.7]])

    assert kl_loss(_padded([[0.3, -0.2]], fill=100.0), target).item() == pytest.approx(
        kl_loss(_padded([[0.3, -0.2]], fill=-50.0), target).item()
    )


@pytest.mark.parametrize("k", range(2, MAX_OPTIONS + 1))
def test_kl_loss_has_finite_gradients_for_every_option_count(k: int) -> None:
    torch.manual_seed(k)
    weights = torch.rand(k)
    weights[0] = 0.0
    target = _padded([(weights / weights.sum()).tolist()])
    logits = _padded([torch.randn(k).tolist()], fill=100.0).requires_grad_(True)

    loss = kl_loss(logits, target)
    (gradient,) = torch.autograd.grad(loss, logits)

    assert torch.isfinite(loss)
    assert torch.isfinite(gradient).all()


def test_uniform_kl_is_log_k_minus_the_target_entropy() -> None:
    target = _padded([[0.25, 0.75], [0.5, 0.5, 0.0]])

    assert uniform_kl(target).tolist() == pytest.approx([KL_A, math.log(3) - math.log(2)])


def test_answer_positions_are_the_sorted_distinct_slots_with_a_column_per_row() -> None:
    positions, columns = answer_positions(torch.tensor([3, 1, 3, 0]))

    assert positions.tolist() == [0, 1, 3]
    assert columns.tolist() == [2, 1, 2, 0]
    assert positions[columns].tolist() == [3, 1, 3, 0]


def test_the_forward_keeps_only_the_answer_positions_and_returns_float32(
    logits_model: ModelFactory,
) -> None:
    logits = torch.zeros(2, 4, 3, dtype=torch.bfloat16)
    logits[0, 3] = torch.tensor([1.0, 2.0, 3.0])
    logits[1, 1] = torch.tensor([4.0, 5.0, 6.0])
    model = logits_model(logits)
    batch = {
        "input_ids": torch.zeros(2, 4, dtype=torch.long),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "answer_index": torch.tensor([3, 1]),
    }

    picked = forward_answer_logits(model, batch)

    assert model.calls[0]["logits_to_keep"].tolist() == [1, 3]
    assert model.calls[0]["use_cache"] is False
    assert picked.dtype is torch.float32
    assert picked.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_rows_of_equal_length_share_one_kept_position(logits_model: ModelFactory) -> None:
    logits = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    model = logits_model(logits)
    batch = {
        "input_ids": torch.zeros(2, 3, dtype=torch.long),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "answer_index": torch.tensor([2, 2]),
    }

    picked = forward_answer_logits(model, batch)

    assert model.calls[0]["logits_to_keep"].tolist() == [2]
    assert picked.tolist() == [logits[0, 2].tolist(), logits[1, 2].tolist()]


def test_a_backbone_that_cannot_restrict_its_head_is_refused(logits_model: ModelFactory) -> None:
    model = logits_model(torch.zeros(1, 2, 3), supports_logits_to_keep=False)
    batch = {
        "input_ids": torch.zeros(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "answer_index": torch.tensor([1]),
    }

    with pytest.raises(RuntimeError, match="does not accept `logits_to_keep`"):
        forward_answer_logits(model, batch)


def test_option_mass_sums_the_vocabulary_probability_of_the_real_options() -> None:
    answer = torch.zeros(1, 8)
    option_ids = torch.tensor([[2, 5, 0, 0]])
    valid = torch.tensor([[True, True, False, False]])

    assert option_mass(answer, option_ids, valid).item() == pytest.approx(2 / 8)


def test_predicted_distribution_normalises_only_over_valid_slots() -> None:
    logits = np.array([[0.0, math.log(3.0), 100.0]])
    valid = np.array([[True, True, False]])

    predicted = predicted_distribution(logits, valid)

    assert predicted.shape == (1, 3)
    assert predicted[0].tolist() == pytest.approx([0.25, 0.75, 0.0])


def test_option_entropy_is_log_k_for_uniform_and_zero_for_one_hot() -> None:
    predicted = np.array([[0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0]])
    valid = np.ones_like(predicted, dtype=bool)

    assert option_entropy(predicted, valid).tolist() == pytest.approx([math.log(4), 0.0])


def test_ordinal_emd_is_zero_for_identical_and_one_for_opposite_ends() -> None:
    predicted = np.array([[0.2, 0.3, 0.5, 0.0], [1.0, 0.0, 0.0, 0.0]])
    target = np.array([[0.2, 0.3, 0.5, TARGET_PAD], [0.0, 0.0, 1.0, TARGET_PAD]])
    valid = target >= 0

    assert ordinal_emd(predicted, target, valid).tolist() == pytest.approx([0.0, 1.0])


def test_compute_metrics_reads_the_tuple_predictions_and_masks_the_sentinels() -> None:
    logits = _padded([[0.0, 0.0]], fill=100.0).numpy()
    mass = np.array([0.4])
    target = _padded([[0.25, 0.75]]).numpy()

    metrics = compute_metrics(
        EvalPrediction(predictions=cast(Any, (logits, mass)), label_ids=target)
    )

    assert set(metrics) == {"kl", "entropy", "emd", "option_mass"}
    assert metrics["kl"] == pytest.approx(KL_A)
    assert metrics["entropy"] == pytest.approx(math.log(2))
    assert metrics["emd"] == pytest.approx(0.25)
    assert metrics["option_mass"] == pytest.approx(0.4)


def test_compute_metrics_rejects_predictions_that_are_not_the_tuple() -> None:
    target = _padded([[0.25, 0.75]]).numpy()

    with pytest.raises(ValueError, match=r"expected \(option_logits, option_mass\)"):
        compute_metrics(EvalPrediction(predictions=target, label_ids=target))


def test_step_epoch_factor_decays_once_per_epoch() -> None:
    factor = step_epoch_factor(0.85, 10)

    assert [factor(step) for step in (0, 9, 10, 20)] == pytest.approx([1.0, 1.0, 0.85, 0.7225])


class _SchedulerHost:
    def __init__(self, lr_gamma: float, steps_per_epoch: int) -> None:
        self.lr_gamma: float | None = lr_gamma
        self.steps_per_epoch = steps_per_epoch
        self.lr_scheduler: Any = None
        self.optimizer: Any = None


def test_create_scheduler_installs_the_per_epoch_step_schedule() -> None:
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    host = _SchedulerHost(0.85, 10)

    scheduler = DistributionalTrainer.create_scheduler(
        cast(DistributionalTrainer, host), 100, optimizer
    )
    learning_rates = []
    for _ in range(21):
        learning_rates.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    assert learning_rates[0] == pytest.approx(1.0)
    assert learning_rates[9] == pytest.approx(1.0)
    assert learning_rates[10] == pytest.approx(0.85)
    assert learning_rates[20] == pytest.approx(0.7225)
    assert host.lr_scheduler is scheduler
    assert getattr(host, "_created_lr_scheduler", False) is True


ModelFactory = Callable[..., Any]


def _batch() -> dict[str, torch.Tensor]:
    return DistributionalCollator(pad_token_id=0)(
        [
            {"input_ids": [5, 6, 7], "option_token_ids": [1, 2], "target_dist": [0.25, 0.75]},
            {"input_ids": [8, 9], "option_token_ids": [1, 2, 3], "target_dist": [1.0, 0.0, 0.0]},
        ]
    )


def _model_for(
    batch: dict[str, torch.Tensor], model_class: ModelFactory, *, letters_win: bool = True
) -> Any:
    logits = torch.zeros(2, batch["input_ids"].shape[1], 4)
    if letters_win:
        logits[1, 1] = torch.tensor([0.0, math.log(2.0), 0.0, 0.0])
    else:
        logits[0, 2] = torch.tensor([100.0, -100.0, -100.0, -100.0])
        logits[1, 1] = torch.tensor([100.0, -100.0, -100.0, -100.0])
    return model_class(logits)


def test_score_batch_forwards_only_the_prompt_and_returns_loss_logits_and_mass(
    logits_model: ModelFactory,
) -> None:
    batch = _batch()
    model = _model_for(batch, logits_model)

    loss, extras = score_batch(model, batch)

    assert set(model.calls[0]) == {"input_ids", "attention_mask", "use_cache", "logits_to_keep"}
    assert model.calls[0]["use_cache"] is False
    assert loss.item() == pytest.approx((KL_A / 2 + KL_B / 3) / 2)
    assert extras["option_logits"].shape == (2, MAX_OPTIONS)
    assert extras["option_mass"].tolist() == pytest.approx([0.5, 0.8])


class _PredictionHost:
    loss_fn = staticmethod(kl_loss)

    def _prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return inputs


def test_prediction_step_returns_the_loss_the_tuple_and_the_targets(
    logits_model: ModelFactory,
) -> None:
    batch = _batch()
    model = _model_for(batch, logits_model)
    host = cast(DistributionalTrainer, _PredictionHost())

    loss, predictions, labels = DistributionalTrainer.prediction_step(
        host, cast(torch.nn.Module, model), batch, False
    )
    loss_only = DistributionalTrainer.prediction_step(
        host, cast(torch.nn.Module, model), batch, True
    )

    assert loss is not None
    assert loss.item() == pytest.approx((KL_A / 2 + KL_B / 3) / 2)
    assert isinstance(predictions, tuple)
    assert predictions[0].shape == (2, MAX_OPTIONS)
    assert predictions[1].shape == (2,)
    assert labels is not None
    assert torch.equal(labels, batch["target_dist"])
    assert loss_only[1] is None
    assert loss_only[2] is None


def test_compute_loss_returns_a_scalar_or_the_pair_with_extras(logits_model: ModelFactory) -> None:
    batch = _batch()
    model = cast(torch.nn.Module, _model_for(batch, logits_model))
    host = cast(DistributionalTrainer, _PredictionHost())

    scalar = DistributionalTrainer.compute_loss(host, model, batch)
    pair = DistributionalTrainer.compute_loss(host, model, batch, return_outputs=True)

    assert isinstance(scalar, torch.Tensor)
    assert scalar.ndim == 0
    assert isinstance(pair, tuple)
    assert set(pair[1]) == {"option_logits", "option_mass"}


class _ExactKlHost(_PredictionHost):
    loss_fn = staticmethod(exact_kl)


def test_the_trainer_scores_with_the_loss_it_was_given(logits_model: ModelFactory) -> None:
    batch = _batch()
    model = cast(torch.nn.Module, _model_for(batch, logits_model))
    host = cast(DistributionalTrainer, _ExactKlHost())

    scalar = DistributionalTrainer.compute_loss(host, model, batch)
    predicted = DistributionalTrainer.prediction_step(host, model, batch, True)

    assert isinstance(scalar, torch.Tensor)
    assert scalar.item() == pytest.approx((KL_A + KL_B) / 2)
    assert predicted[0] is not None
    assert predicted[0].item() == pytest.approx((KL_A + KL_B) / 2)
    assert score_batch(model, batch, exact_kl)[0].item() == pytest.approx((KL_A + KL_B) / 2)


class _FakeTrainer:
    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        self.batch = batch

    def get_train_dataloader(self) -> list[dict[str, torch.Tensor]]:
        return [self.batch]


def test_the_base_check_reports_kl_ratio_and_mass_and_restores_training_mode(
    logits_model: ModelFactory,
) -> None:
    batch = _batch()
    model = _model_for(batch, logits_model)

    kl, ratio, mass = assert_base_elicits_options(_FakeTrainer(batch), model)

    assert kl == pytest.approx((KL_A / 2 + KL_B / 3) / 2)
    assert ratio == pytest.approx(kl / ((KL_A / 2 + (math.log(3) - 0.0) / 3) / 2))
    assert mass == pytest.approx((0.5 + 0.8) / 2)
    assert model.training is True


def test_the_base_check_fails_when_the_letters_get_no_mass(logits_model: ModelFactory) -> None:
    batch = _batch()
    model = _model_for(batch, logits_model, letters_win=False)

    with pytest.raises(RuntimeError, match="mass on the option letters"):
        assert_base_elicits_options(_FakeTrainer(batch), model)
    assert model.training is True


def test_the_base_check_names_the_stub_it_was_given(logits_model: ModelFactory) -> None:
    batch = _batch()
    model = _model_for(batch, logits_model, letters_win=False)

    with pytest.raises(RuntimeError, match="after 'Answer:'"):
        assert_base_elicits_options(_FakeTrainer(batch), model, stub="Answer:")


def _write_splits(raw_dir: Path, raw_survey_record: RecordFactory) -> None:
    raw_dir.mkdir()
    rows = {
        "train": [
            raw_survey_record(record_id="46", country="Germany"),
            raw_survey_record(record_id="57", country="Germany"),
            raw_survey_record(record_id="1", country="Germany"),
        ],
        "valid": [
            raw_survey_record(record_id="171", country="Germany", data_type="valid"),
            raw_survey_record(record_id="164", country="Germany", data_type="valid"),
        ],
        "test": [raw_survey_record(record_id="240", country="Mexico", data_type="test_7")],
    }
    for split, name in SPLIT_FILES.items():
        (raw_dir / name).write_text(json.dumps(rows[split]))


def test_load_training_splits_follows_the_protocol_unless_asked_to_filter(
    tmp_path: Path, survey_tokenizer: TokenizerFactory, raw_survey_record: RecordFactory
) -> None:
    raw_dir = tmp_path / "raw"
    _write_splits(raw_dir, raw_survey_record)
    encoder = PromptEncoder.build(survey_tokenizer(), max_length=256)

    train, valid, stats = load_training_splits(
        encoder, raw_dir=raw_dir, exclude_items=False, debug=False
    )
    filtered_train, filtered_valid, filtered_stats = load_training_splits(
        encoder, raw_dir=raw_dir, exclude_items=True, debug=False
    )

    assert (train.num_rows, valid.num_rows) == (3, 2)
    assert stats["dropped_rows"] == 0
    assert stats["excluded_items"] == ""
    assert stats["evaluation_item_rows_in_train"] == "46=1,57=1,171=0,240=0"
    assert (filtered_train.num_rows, filtered_valid.num_rows) == (1, 1)
    assert filtered_stats["dropped_rows"] == 3
    assert filtered_stats["excluded_items"] == "46,57,171,240"
    assert filtered_stats["evaluation_item_rows_in_train"] == "46=0,57=0,171=0,240=0"


def test_debug_keeps_only_a_small_slice_of_each_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    survey_tokenizer: TokenizerFactory,
    raw_survey_record: RecordFactory,
) -> None:
    raw_dir = tmp_path / "raw"
    _write_splits(raw_dir, raw_survey_record)
    monkeypatch.setattr(train_distributional, "DEBUG_TRAIN_ROWS", 2)
    monkeypatch.setattr(train_distributional, "DEBUG_VALID_ROWS", 1)
    encoder = PromptEncoder.build(survey_tokenizer(), max_length=256)

    train, valid, stats = load_training_splits(
        encoder, raw_dir=raw_dir, exclude_items=False, debug=True
    )

    assert (train.num_rows, valid.num_rows) == (2, 1)
    assert stats["train_rows"] == 2


def _load(path: Path) -> dict[str, Any]:
    with open(path) as handle:
        return dict(yaml.safe_load(handle))


@pytest.fixture(params=DISTRIBUTIONAL_CONFIGS, ids=lambda p: p.stem)
def distributional_config(request: pytest.FixtureRequest) -> tuple[Path, dict[str, Any]]:
    path = Path(request.param)
    return path, _load(path)


def test_the_deliverable_backbone_has_a_distributional_config() -> None:
    assert CONFIG_DIR / "qwen3_vl_2b_distributional.yaml" in DISTRIBUTIONAL_CONFIGS


def test_a_distributional_config_names_its_track_and_pins_the_backbone_key(
    distributional_config: tuple[Path, dict[str, Any]],
) -> None:
    path, cfg = distributional_config

    assert cfg["condition"] == "distributional"
    assert cfg["data"]["format"] == "distributional"
    assert isinstance(cfg["data"]["exclude_evaluation_items"], bool)
    assert cfg["model"]["key"] == path.stem.removesuffix("_distributional")
    assert "cultures" not in cfg
    assert "n_folds" not in cfg
    assert cfg["mlflow"]["experiment"] == "culture_mllm_distributional_training"


def test_a_distributional_config_loads_the_backbone_exactly_as_the_wvs_config_does(
    distributional_config: tuple[Path, dict[str, Any]],
) -> None:
    _, cfg = distributional_config
    base = _load(CONFIG_DIR / f"{cfg['model']['key']}.yaml")

    assert {key: value for key, value in cfg["model"].items() if key != "key"} == base["model"]


def test_a_distributional_config_follows_the_reference_recipe(
    distributional_config: tuple[Path, dict[str, Any]],
) -> None:
    _, cfg = distributional_config
    train_cfg = cfg["training"]

    assert cfg["lora"] == {
        "r": 8,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
    }
    assert train_cfg["learning_rate"] == pytest.approx(1e-4)
    assert train_cfg["max_epochs"] == 6
    assert train_cfg["batch_size"] * train_cfg["gradient_accumulation"] == 8
    assert train_cfg["lr_schedule"] == "step_epoch"
    assert train_cfg["lr_gamma"] == pytest.approx(0.85)
    assert train_cfg["warmup_steps"] == 0
    assert train_cfg["max_seq_len"] >= 256
    assert train_cfg["save_steps"] % train_cfg["eval_steps"] == 0
