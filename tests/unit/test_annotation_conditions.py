"""Pin the annotation condition contract.

A silently renamed condition, a checkpoint directory that stops matching the
Stage-2 layout on disk, or an adapter that quietly falls back to the raw base
model would all produce annotation runs that look valid but answer a different
experimental question.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.annotation.conditions import (
    CONDITIONS,
    checkpoint_directory,
    normalize_condition,
)
from src.annotation.config import AnnotatorSettings, annotation_seed
from src.annotation.nodes.annotator import (
    AdapterCheckpointError,
    _adapter_path,
    _parse_output_with_strategy,
    validate_inference_assets,
)


def test_condition_registry_and_legacy_normalization() -> None:
    assert CONDITIONS == ("inference_only", "wvs_cultural")
    assert normalize_condition("cultural") == "wvs_cultural"
    assert normalize_condition(" WVS_CULTURAL ") == "wvs_cultural"
    with pytest.raises(ValueError, match="Unknown annotation condition"):
        normalize_condition("baseline")


def test_checkpoint_layout_maps_wvs_to_legacy_directory(tmp_path: Path) -> None:
    assert checkpoint_directory(tmp_path, "arabic", "m", "wvs_cultural") == (
        tmp_path / "arabic" / "m" / "cultural"
    )
    assert checkpoint_directory(tmp_path, "arabic", "m", "cultural") == (
        tmp_path / "arabic" / "m" / "cultural"
    )
    assert checkpoint_directory(tmp_path, "arabic", "m", "inference_only") is None


def test_trained_adapter_never_falls_back_to_raw_model(tmp_path: Path) -> None:
    settings = AnnotatorSettings(
        checkpoints_dir=tmp_path,
        model_id_map={},
        hf_model_id_map={"test_model": "org/base"},
    )
    with pytest.raises(AdapterCheckpointError, match="Missing adapter"):
        validate_inference_assets("arabic", "test_model", "wvs_cultural", settings)

    adapter = tmp_path / "arabic" / "test_model" / "cultural"
    adapter.mkdir(parents=True)
    (adapter / "adapters.safetensors").touch()
    with pytest.raises(AdapterCheckpointError, match="incompatible with the hf backend"):
        validate_inference_assets("arabic", "test_model", "wvs_cultural", settings)

    (adapter / "adapter_model.safetensors").touch()
    backend, path, identity = validate_inference_assets(
        "arabic", "test_model", "wvs_cultural", settings
    )
    assert backend == "hf"
    assert path == adapter
    assert "adapter_model.safetensors" in identity
    assert identity.endswith(
        "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )

    backend, path, identity = validate_inference_assets(
        "inference_only", "test_model", "inference_only", settings
    )
    assert backend == "hf"
    assert path is None
    assert identity == "base::org/base"


def test_adapter_path_requires_supported_artifact(tmp_path: Path) -> None:
    path = tmp_path / "arabic" / "m" / "cultural"
    path.mkdir(parents=True)
    with pytest.raises(AdapterCheckpointError, match="no supported weights"):
        _adapter_path(
            "arabic",
            "m",
            "wvs_cultural",
            checkpoints_dir=tmp_path,
        )


def test_seed_is_stable_and_condition_independent_by_construction() -> None:
    first = annotation_seed(42, "model", "image-1", 1)
    assert first == annotation_seed(42, "model", "image-1", 1)
    assert first != annotation_seed(42, "model", "image-1", 2)
    assert first != annotation_seed(42, "other-model", "image-1", 1)
    assert 0 <= first < 2**31 - 1


def test_invalid_unstructured_parse_is_a_failure_not_synthetic_neutral() -> None:
    parsed, strategy = _parse_output_with_strategy(
        "This seems positive overall but I did not return JSON."
    )
    assert parsed is None
    assert strategy == "invalid"

    parsed, strategy = _parse_output_with_strategy(
        'prefix {"sentiment": "positive", "caption": "A sunny city street", '
        '"justification": "The scene is bright and welcoming.", "tags": ["sunny"]} suffix'
    )
    assert parsed is not None
    assert parsed["sentiment"] == 4
    assert strategy == "extracted_json"
