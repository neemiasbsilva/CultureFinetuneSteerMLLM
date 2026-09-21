"""Pin the run harness the two training tracks share, so relocating it moved no behaviour."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
import pytest
import torch
from peft import LoraConfig, TaskType

from src.training import common, train_hf
from src.training.common import (
    EarlyStopCallback,
    configure_mlflow,
    early_stopping_callback,
    find_last_checkpoint,
    resume_state,
    trainer_arguments,
    wrap_lora,
)
from src.training.early_stopping import EarlyStopping

TRAIN_CFG: dict[str, Any] = {
    "batch_size": 4,
    "gradient_accumulation": 4,
    "max_epochs": 250,
    "learning_rate": 2e-4,
    "lr_schedule": "cosine",
    "warmup_steps": 50,
    "early_stopping_patience": 15,
    "early_stopping_min_epochs": 20,
    "max_seq_len": 512,
    "logging_steps": 25,
    "eval_steps": 100,
    "save_steps": 100,
    "max_grad_norm": 0.3,
}


@pytest.mark.parametrize(
    "name", ["default_exclude_modules", "get_model_key", "checkpoint_output_dir", "load_config"]
)
def test_train_hf_re_exports_the_shared_helpers_by_identity(name: str) -> None:
    assert getattr(train_hf, name) is getattr(common, name)


def test_checkpoint_layout_is_culture_then_model_then_problem() -> None:
    assert common.checkpoint_output_dir("global", "qwen3_vl_2b", "distributional") == Path(
        "checkpoints/global/qwen3_vl_2b/distributional"
    )


def test_trainer_arguments_reproduce_the_pre_refactor_keyword_set_with_validation() -> None:
    kwargs = trainer_arguments(
        TRAIN_CFG, output_dir="out", run_name="run", has_validation=True, debug=False, use_bf16=True
    )
    assert kwargs == {
        "output_dir": "out",
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 250,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 50,
        "bf16": True,
        "fp16": False,
        "optim": "adamw_torch",
        "eval_strategy": "steps",
        "eval_steps": 100,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": None,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "mlflow",
        "logging_steps": 25,
        "max_grad_norm": 0.3,
        "run_name": "run",
    }


def test_trainer_arguments_without_validation_turn_off_evaluation_and_best_model_selection() -> (
    None
):
    kwargs = trainer_arguments(
        TRAIN_CFG,
        output_dir="out",
        run_name="run",
        has_validation=False,
        debug=False,
        use_bf16=False,
    )
    assert kwargs["eval_strategy"] == "no"
    assert kwargs["eval_steps"] is None
    assert kwargs["load_best_model_at_end"] is False
    assert kwargs["metric_for_best_model"] is None
    assert kwargs["bf16"] is False


def test_debug_runs_train_for_exactly_one_epoch() -> None:
    kwargs = trainer_arguments(
        TRAIN_CFG, output_dir="out", run_name="run", has_validation=True, debug=True, use_bf16=True
    )
    assert kwargs["num_train_epochs"] == 1


def test_the_schedule_defaults_to_cosine_and_save_total_limit_passes_through() -> None:
    cfg = {key: value for key, value in TRAIN_CFG.items() if key != "lr_schedule"}
    cfg["save_total_limit"] = 3
    kwargs = trainer_arguments(
        cfg, output_dir="out", run_name="run", has_validation=True, debug=False, use_bf16=True
    )
    assert kwargs["lr_scheduler_type"] == "cosine"
    assert kwargs["save_total_limit"] == 3


class _FakePeftModel:
    def __init__(self, params: list[torch.nn.Parameter], *, loaded_in_4bit: bool) -> None:
        self._params = params
        self.is_loaded_in_4bit = loaded_in_4bit
        self.input_grads_enabled = False

    def parameters(self) -> list[torch.nn.Parameter]:
        return self._params

    def enable_input_require_grads(self) -> None:
        self.input_grads_enabled = True


def _params() -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    trainable = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32), requires_grad=True)
    frozen = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32), requires_grad=False)
    return trainable, frozen


def _lora_config() -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )


def _install_fake_peft(monkeypatch: pytest.MonkeyPatch, peft_model: _FakePeftModel) -> list[Any]:
    calls: list[Any] = []

    def fake_get_peft_model(model: Any, config: Any) -> _FakePeftModel:
        calls.append((model, config))
        return peft_model

    monkeypatch.setattr(common, "get_peft_model", fake_get_peft_model)
    return calls


def test_wrap_lora_casts_only_trainable_parameters_to_bf16_on_a_quantized_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainable, frozen = _params()
    peft_model = _FakePeftModel([trainable, frozen], loaded_in_4bit=True)
    calls = _install_fake_peft(monkeypatch, peft_model)
    lora_config = _lora_config()

    wrapped = wrap_lora("base", lora_config, quantization="4bit", gradient_checkpointing=False)

    assert wrapped is peft_model
    assert calls == [("base", lora_config)]
    assert trainable.dtype is torch.bfloat16
    assert frozen.dtype is torch.float32


def test_wrap_lora_trusts_the_loaded_in_4bit_attribute_when_no_strategy_is_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainable, frozen = _params()
    _install_fake_peft(monkeypatch, _FakePeftModel([trainable, frozen], loaded_in_4bit=True))

    wrap_lora("base", _lora_config(), quantization=None, gradient_checkpointing=False)

    assert trainable.dtype is torch.bfloat16


def test_wrap_lora_leaves_parameter_dtypes_alone_on_an_unquantized_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainable, frozen = _params()
    _install_fake_peft(monkeypatch, _FakePeftModel([trainable, frozen], loaded_in_4bit=False))

    wrap_lora("base", _lora_config(), quantization=None, gradient_checkpointing=False)

    assert trainable.dtype is torch.float32
    assert frozen.dtype is torch.float32


@pytest.mark.parametrize("gradient_checkpointing", [True, False])
def test_wrap_lora_enables_input_gradients_only_with_gradient_checkpointing(
    monkeypatch: pytest.MonkeyPatch, gradient_checkpointing: bool
) -> None:
    peft_model = _FakePeftModel(list(_params()), loaded_in_4bit=False)
    _install_fake_peft(monkeypatch, peft_model)

    wrap_lora(
        "base", _lora_config(), quantization=None, gradient_checkpointing=gradient_checkpointing
    )

    assert peft_model.input_grads_enabled is gradient_checkpointing


def test_find_last_checkpoint_orders_numerically_not_lexically(tmp_path: Path) -> None:
    for step in (100, 2000, 900):
        (tmp_path / f"checkpoint-{step}").mkdir()

    assert find_last_checkpoint(str(tmp_path)) == str(tmp_path / "checkpoint-2000")


def test_find_last_checkpoint_is_none_without_checkpoints(tmp_path: Path) -> None:
    assert find_last_checkpoint(str(tmp_path)) is None


class _RunInfo:
    def __init__(self, status: str) -> None:
        self.status = status


class _Run:
    def __init__(self, status: str) -> None:
        self.info = _RunInfo(status)


def _client_class(runs: dict[str, str]) -> tuple[type, list[tuple[str, str]]]:
    updates: list[tuple[str, str]] = []

    class _FakeMlflowClient:
        def get_run(self, run_id: str) -> _Run:
            if run_id not in runs:
                raise KeyError(run_id)
            return _Run(runs[run_id])

        def update_run(self, run_id: str, status: str) -> None:
            updates.append((run_id, status))
            runs[run_id] = status

    return _FakeMlflowClient, updates


def test_resume_state_reopens_the_saved_run_and_forces_it_back_to_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mlflow_run_id.txt").write_text("abc123")
    (tmp_path / "checkpoint-300").mkdir()
    client_class, updates = _client_class({"abc123": "FAILED"})
    monkeypatch.setattr("mlflow.tracking.MlflowClient", client_class)

    state = resume_state(str(tmp_path), "run")

    assert state.run_id == "abc123"
    assert state.is_new is False
    assert state.last_checkpoint == str(tmp_path / "checkpoint-300")
    assert updates == [("abc123", "RUNNING")]


def test_resume_state_leaves_a_running_run_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mlflow_run_id.txt").write_text("abc123")
    client_class, updates = _client_class({"abc123": "RUNNING"})
    monkeypatch.setattr("mlflow.tracking.MlflowClient", client_class)

    state = resume_state(str(tmp_path), "run")

    assert state.run_id == "abc123"
    assert updates == []


def test_resume_state_starts_a_new_run_when_the_saved_id_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mlflow_run_id.txt").write_text("gone")
    client_class, _ = _client_class({})
    monkeypatch.setattr("mlflow.tracking.MlflowClient", client_class)

    state = resume_state(str(tmp_path), "run")

    assert state.run_id is None
    assert state.is_new is True
    assert state.last_checkpoint is None


def test_resume_state_starts_fresh_without_a_saved_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client_class, _ = _client_class({})
    monkeypatch.setattr("mlflow.tracking.MlflowClient", client_class)

    assert resume_state(str(tmp_path), "run") == common.ResumeState(None, None, True)


class _State:
    def __init__(self, epoch: float, global_step: int) -> None:
        self.epoch = epoch
        self.global_step = global_step


class _Control:
    def __init__(self) -> None:
        self.should_training_stop = False


def _evaluate(callback: EarlyStopCallback, losses: list[float | None]) -> list[bool]:
    stops: list[bool] = []
    for step, loss in enumerate(losses, start=1):
        control = _Control()
        metrics = {} if loss is None else {"eval_loss": loss, "eval_entropy": 1.0}
        callback.on_evaluate(None, _State(float(step), step * 100), control, metrics)
        stops.append(control.should_training_stop)
    return stops


def test_the_early_stop_callback_stops_after_patience_non_improving_evaluations() -> None:
    stopper = EarlyStopping(patience=2, min_delta=0.0, min_epochs=0, ema_alpha=1.0, monitor="loss")
    callback = EarlyStopCallback(stopper)

    assert _evaluate(callback, [1.0, 1.5, 1.5]) == [False, False, True]


def test_the_early_stop_callback_ignores_evaluations_without_a_loss() -> None:
    stopper = EarlyStopping(patience=1, min_delta=0.0, min_epochs=0, ema_alpha=1.0, monitor="loss")
    callback = EarlyStopCallback(stopper)

    assert _evaluate(callback, [1.0, None, None]) == [False, False, False]
    assert stopper.counter == 0


def test_early_stopping_callback_reads_patience_and_warmup_from_the_config() -> None:
    callback = early_stopping_callback(TRAIN_CFG)

    assert callback.stopper.patience == 15
    assert callback.stopper.min_epochs == 20
    assert callback.stopper.monitor == "loss"


def test_early_stopping_callback_defaults_the_warmup_to_ten_epochs() -> None:
    cfg = {key: value for key, value in TRAIN_CFG.items() if key != "early_stopping_min_epochs"}

    assert early_stopping_callback(cfg).stopper.min_epochs == 10


def test_configure_mlflow_points_the_client_and_the_environment_at_the_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: calls.append(("uri", uri)))
    monkeypatch.setattr(mlflow, "set_experiment", lambda name: calls.append(("exp", name)))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "stale")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "stale")

    result = configure_mlflow({"mlflow": {"experiment": "exp", "tracking_uri": "http://x:1"}})

    assert result == ("http://x:1", "exp")
    assert calls == [("uri", "http://x:1"), ("exp", "exp")]
    assert os.environ["MLFLOW_TRACKING_URI"] == "http://x:1"
    assert os.environ["MLFLOW_EXPERIMENT_NAME"] == "exp"


def test_configure_mlflow_defaults_the_tracking_uri_to_the_local_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda name: None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "stale")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "stale")

    assert configure_mlflow({"mlflow": {"experiment": "exp"}}) == ("http://127.0.0.1:5000", "exp")
