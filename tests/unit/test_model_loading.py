"""Pin the one place that decides how base weights reach the device.

Every adapter in the paper is trained on top of whatever `build_base_model`
handed back, and the ways this goes wrong are all silent: a `dtype` that reads
as declared but resolves against absent checkpoint metadata, a pre-quantized
release loaded as if it were BF16, a `None` quantization config that overwrites
the checkpoint's own, or a `device_map` that shards a model the trainer assumed
was on one visible device.  None of those raise — they just train something
other than what the config describes, so they are asserted here instead.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from src.utils.model_loading import (
    QUANTIZATION_STRATEGIES,
    ModelConfigError,
    build_base_model,
    checkpoint_quant_method,
    resolve_dtype,
    resolve_quantization,
    validate_base_model_config,
)


class _FakeModel:
    def __init__(self) -> None:
        self.moved_to: str | None = None

    def to(self, device: str) -> _FakeModel:
        self.moved_to = device
        return self


class _RecordingAutoClass:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.model = _FakeModel()

    def from_pretrained(self, model_id: str, **kwargs: Any) -> _FakeModel:
        self.calls.append((model_id, kwargs))
        return self.model

    @property
    def last_kwargs(self) -> dict[str, Any]:
        return self.calls[-1][1]


class _RecordingQuantMethod:
    def __init__(self, method: str | None = None) -> None:
        self.method = method
        self.inspected: list[str] = []

    def __call__(self, model_id: str) -> str | None:
        self.inspected.append(model_id)
        return self.method


class _EmbeddedQuantConfig:
    def __init__(self, quant_method: str) -> None:
        self.quant_method = quant_method


class _FakeCheckpointConfig:
    def __init__(self, quantization_config: Any) -> None:
        self.quantization_config = quantization_config


class _ConfigWithoutQuantizationSection:
    pass


class _FakeAutoConfig:
    def __init__(self, config: Any = None, error: Exception | None = None) -> None:
        self.config = config
        self.error = error
        self.inspected: list[str] = []
        self.calls: list[dict[str, Any]] = []

    def from_pretrained(self, model_id: str, **kwargs: Any) -> Any:
        self.inspected.append(model_id)
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.config


@pytest.mark.parametrize(
    "model_cfg",
    [{}, {"quantization": None}, {"quantization": ""}, {"quantization": "none"}],
)
def test_absent_empty_and_none_quantization_all_mean_an_immediate_load(
    model_cfg: dict[str, Any],
) -> None:
    assert resolve_quantization(model_cfg) is None
    assert resolve_quantization(model_cfg, {}) is None


def test_declared_four_bit_strategy_survives_resolution_unchanged() -> None:
    assert resolve_quantization({"quantization": "4bit"}) == "4bit"
    assert resolve_quantization({}, {"quantization": "4bit"}) == "4bit"


def test_training_section_quantization_overrides_the_model_section() -> None:
    assert resolve_quantization({"quantization": None}, {"quantization": "4bit"}) == "4bit"
    assert resolve_quantization({"quantization": "4bit"}, {"quantization": None}) is None
    assert resolve_quantization({"quantization": "4bit"}, {"quantization": "none"}) is None


def test_model_section_quantization_is_used_when_the_training_section_omits_the_key() -> None:
    assert resolve_quantization({"quantization": "4bit"}, {"learning_rate": 1e-4}) == "4bit"
    assert resolve_quantization({"quantization": "4bit"}, None) == "4bit"


@pytest.mark.parametrize("declared", ["8bit", "nf4", "gptq", "4BIT", "int8"])
def test_unknown_quantization_strategy_is_refused_and_names_what_is_accepted(
    declared: str,
) -> None:
    with pytest.raises(ModelConfigError, match=f"Unknown quantization strategy {declared!r}"):
        resolve_quantization({"quantization": declared})
    with pytest.raises(ModelConfigError, match="Expected one of: '4bit'"):
        resolve_quantization({"quantization": declared})


def test_only_none_and_four_bit_are_advertised_as_supported_strategies() -> None:
    assert QUANTIZATION_STRATEGIES == (None, "4bit")


def test_model_config_error_still_reads_as_a_value_error_to_existing_callers() -> None:
    assert issubclass(ModelConfigError, ValueError)
    with pytest.raises(ValueError, match="Unknown quantization strategy"):
        resolve_quantization({"quantization": "8bit"})


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
        ("float32", torch.float32),
    ],
)
def test_declared_dtype_string_maps_onto_the_matching_torch_dtype(
    declared: str, expected: torch.dtype
) -> None:
    assert resolve_dtype({"dtype": declared}) is expected


def test_auto_dtype_is_refused_because_composite_configs_carry_no_checkpoint_dtype() -> None:
    with pytest.raises(ModelConfigError, match=r'"auto" is not accepted'):
        resolve_dtype({"dtype": "auto"})


def test_missing_dtype_is_refused_rather_than_silently_defaulting() -> None:
    with pytest.raises(ModelConfigError, match=r"model\.dtype is required"):
        resolve_dtype({})
    with pytest.raises(ModelConfigError, match=r"model\.dtype is required"):
        resolve_dtype({"dtype": None})


@pytest.mark.parametrize("declared", ["fp8", "bf16", "float64", "int4"])
def test_unrecognised_dtype_string_is_refused_and_lists_the_known_dtypes(declared: str) -> None:
    with pytest.raises(ModelConfigError, match=f"Unknown dtype {declared!r}"):
        resolve_dtype({"dtype": declared})
    with pytest.raises(ModelConfigError, match="bfloat16, float16, float32"):
        resolve_dtype({"dtype": declared})


@pytest.mark.parametrize(
    ("train_cfg", "expected"),
    [(None, None), ({"quantization": "4bit"}, "4bit"), ({"quantization": "none"}, None)],
)
def test_unquantized_checkpoint_lets_the_declared_strategy_through(
    monkeypatch: pytest.MonkeyPatch, train_cfg: dict[str, Any] | None, expected: str | None
) -> None:
    inspector = _RecordingQuantMethod(None)
    monkeypatch.setattr("src.utils.model_loading.checkpoint_quant_method", inspector)
    strategy = validate_base_model_config("org/base-bf16", {"dtype": "bfloat16"}, train_cfg)
    assert strategy == expected
    assert inspector.inspected == ["org/base-bf16"]


@pytest.mark.parametrize("embedded", ["fp8", "bitsandbytes", "awq", "gptq"])
@pytest.mark.parametrize("train_cfg", [None, {"quantization": "4bit"}])
def test_prequantized_checkpoint_is_refused_under_every_declared_strategy(
    monkeypatch: pytest.MonkeyPatch, embedded: str, train_cfg: dict[str, Any] | None
) -> None:
    monkeypatch.setattr(
        "src.utils.model_loading.checkpoint_quant_method", _RecordingQuantMethod(embedded)
    )
    with pytest.raises(ModelConfigError, match=f"pre-quantized with quant_method={embedded!r}"):
        validate_base_model_config("org/base-fp8", {"dtype": "bfloat16"}, train_cfg)


def test_dtype_is_validated_before_the_checkpoint_is_ever_inspected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = _RecordingQuantMethod(None)
    monkeypatch.setattr("src.utils.model_loading.checkpoint_quant_method", inspector)
    with pytest.raises(ModelConfigError, match=r'"auto" is not accepted'):
        validate_base_model_config("org/base-bf16", {"dtype": "auto"})
    with pytest.raises(ModelConfigError, match="Unknown quantization strategy"):
        validate_base_model_config("org/base-bf16", {"dtype": "bfloat16"}, {"quantization": "8bit"})
    assert inspector.inspected == []


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (_FakeCheckpointConfig({"quant_method": "fp8"}), "fp8"),
        (_FakeCheckpointConfig(_EmbeddedQuantConfig("awq")), "awq"),
        (_FakeCheckpointConfig({}), None),
        (_FakeCheckpointConfig(None), None),
        (_ConfigWithoutQuantizationSection(), None),
    ],
)
def test_embedded_quantization_metadata_is_read_from_dict_and_object_configs(
    monkeypatch: pytest.MonkeyPatch, config: Any, expected: str | None
) -> None:
    monkeypatch.setattr("src.utils.model_loading.AutoConfig", _FakeAutoConfig(config))
    assert checkpoint_quant_method("org/base") == expected


def test_checkpoint_inspection_reads_config_metadata_only_and_trusts_remote_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeAutoConfig(_FakeCheckpointConfig({"quant_method": "fp8"}))
    monkeypatch.setattr("src.utils.model_loading.AutoConfig", fake)
    checkpoint_quant_method("org/base-fp8")
    assert fake.inspected == ["org/base-fp8"]
    assert fake.calls == [{"trust_remote_code": True}]


def test_unreadable_checkpoint_config_reports_no_quantization_instead_of_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeAutoConfig(error=OSError("offline: org/base is not cached"))
    monkeypatch.setattr("src.utils.model_loading.AutoConfig", fake)
    assert checkpoint_quant_method("org/base") is None
    assert fake.inspected == ["org/base"]


def test_cuda_load_pins_every_module_to_the_single_visible_device() -> None:
    auto_class = _RecordingAutoClass()
    model = build_base_model(
        "org/base",
        auto_class=auto_class,
        quantization=None,
        dtype=torch.bfloat16,
        device="cuda",
    )
    assert auto_class.calls[0][0] == "org/base"
    assert auto_class.last_kwargs["device_map"] == {"": "cuda"}
    assert auto_class.last_kwargs["dtype"] is torch.bfloat16
    assert auto_class.last_kwargs["trust_remote_code"] is True
    assert model is auto_class.model
    assert model.moved_to is None


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_non_cuda_load_leaves_device_map_unset_and_moves_the_model_afterwards(
    device: str,
) -> None:
    auto_class = _RecordingAutoClass()
    model = build_base_model(
        "org/base",
        auto_class=auto_class,
        quantization=None,
        dtype=torch.float32,
        device=device,
    )
    assert auto_class.last_kwargs["device_map"] is None
    assert auto_class.last_kwargs["dtype"] is torch.float32
    assert model.moved_to == device


@pytest.mark.parametrize("device", ["cuda", "cpu", "mps"])
def test_unquantized_loads_never_pass_a_quantization_config_kwarg(device: str) -> None:
    auto_class = _RecordingAutoClass()
    build_base_model(
        "org/base",
        auto_class=auto_class,
        quantization=None,
        dtype=torch.bfloat16,
        device=device,
    )
    assert "quantization_config" not in auto_class.last_kwargs


def test_four_bit_on_cuda_passes_a_double_quantised_nf4_bitsandbytes_config() -> None:
    auto_class = _RecordingAutoClass()
    model = build_base_model(
        "org/base",
        auto_class=auto_class,
        quantization="4bit",
        dtype=torch.bfloat16,
        device="cuda",
    )
    quant_config = auto_class.last_kwargs["quantization_config"]
    assert quant_config.load_in_4bit is True
    assert quant_config.bnb_4bit_quant_type == "nf4"
    assert quant_config.bnb_4bit_compute_dtype is torch.bfloat16
    assert quant_config.bnb_4bit_use_double_quant is True
    assert auto_class.last_kwargs["device_map"] == {"": "cuda"}
    assert model.moved_to is None


def test_four_bit_cuda_load_omits_the_top_level_dtype_kwarg_entirely() -> None:
    auto_class = _RecordingAutoClass()
    build_base_model(
        "org/base",
        auto_class=auto_class,
        quantization="4bit",
        dtype=torch.bfloat16,
        device="cuda",
    )
    assert "dtype" not in auto_class.last_kwargs
    assert auto_class.last_kwargs["quantization_config"].bnb_4bit_compute_dtype is torch.bfloat16


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_four_bit_off_cuda_falls_back_to_a_full_precision_load(device: str) -> None:
    auto_class = _RecordingAutoClass()
    model = build_base_model(
        "org/base",
        auto_class=auto_class,
        quantization="4bit",
        dtype=torch.float32,
        device=device,
    )
    assert "quantization_config" not in auto_class.last_kwargs
    assert auto_class.last_kwargs["dtype"] is torch.float32
    assert auto_class.last_kwargs["device_map"] is None
    assert model.moved_to == device


@pytest.mark.parametrize(
    ("quantization", "device"),
    [(None, "cuda"), (None, "cpu"), ("4bit", "cuda"), ("4bit", "cpu")],
)
def test_attention_implementation_is_forwarded_only_when_it_is_declared(
    quantization: str | None, device: str
) -> None:
    default_class = _RecordingAutoClass()
    build_base_model(
        "org/base",
        auto_class=default_class,
        quantization=quantization,
        dtype=torch.bfloat16,
        device=device,
    )
    assert "attn_implementation" not in default_class.last_kwargs

    eager_class = _RecordingAutoClass()
    build_base_model(
        "org/base",
        auto_class=eager_class,
        quantization=quantization,
        dtype=torch.bfloat16,
        device=device,
        attn_implementation="eager",
    )
    assert eager_class.last_kwargs["attn_implementation"] == "eager"


@pytest.mark.parametrize("quantization", [None, "4bit"])
def test_trust_remote_code_reaches_from_pretrained_on_both_load_paths(
    quantization: str | None,
) -> None:
    auto_class = _RecordingAutoClass()
    build_base_model(
        "org/base",
        auto_class=auto_class,
        quantization=quantization,
        dtype=torch.bfloat16,
        device="cuda",
        trust_remote_code=False,
    )
    assert auto_class.last_kwargs["trust_remote_code"] is False
