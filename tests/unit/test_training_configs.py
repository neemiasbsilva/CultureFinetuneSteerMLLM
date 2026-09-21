"""Pin the shipped training configs against the loader that reads them."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from src.training.train_hf import default_exclude_modules, get_model_key
from src.utils.model_loading import resolve_dtype, resolve_modality, resolve_quantization

CONFIG_DIR = Path("configs")
WVS_CONFIGS = sorted(p for p in CONFIG_DIR.glob("*.yaml") if not p.stem.endswith("_yfcc"))
GERMAN_ONLY_MODELS = ("qwen3_vl_2b", "llama3_2_3b")


def _load(path: Path) -> dict[str, Any]:
    with open(path) as handle:
        return dict(yaml.safe_load(handle))


@pytest.fixture(params=WVS_CONFIGS, ids=lambda p: p.stem)
def wvs_config(request: pytest.FixtureRequest) -> tuple[Path, dict[str, Any]]:
    path = Path(request.param)
    return path, _load(path)


def test_every_wvs_config_declares_a_loadable_model_section(
    wvs_config: tuple[Path, dict[str, Any]],
) -> None:
    _, cfg = wvs_config
    model_cfg, train_cfg = cfg["model"], cfg["training"]
    assert model_cfg["id"]
    resolve_dtype(model_cfg)
    resolve_modality(model_cfg)
    resolve_quantization(model_cfg, train_cfg)


def test_an_image_budget_is_declared_by_exactly_the_configs_that_see_images(
    wvs_config: tuple[Path, dict[str, Any]],
) -> None:
    _, cfg = wvs_config
    model_cfg = cfg["model"]
    has_budget = model_cfg.get("image_max_pixels") is not None
    assert has_budget is (resolve_modality(model_cfg) == "vision_text")


def test_a_configured_chat_template_is_shipped_alongside_the_config(
    wvs_config: tuple[Path, dict[str, Any]],
) -> None:
    _, cfg = wvs_config
    template = cfg["model"].get("chat_template")
    if template is None:
        return
    path = Path(template)
    assert path.is_file(), f"chat_template {path} is not in the repository"
    assert path.read_text().strip()


def test_text_only_configs_exclude_nothing_because_they_are_all_language_path(
    wvs_config: tuple[Path, dict[str, Any]],
) -> None:
    _, cfg = wvs_config
    modality = resolve_modality(cfg["model"])
    declared = cfg["lora"].get("exclude_modules", default_exclude_modules(modality))
    if modality == "text":
        assert declared is None
    else:
        assert declared and "vision" in declared


@pytest.mark.parametrize("model_key", GERMAN_ONLY_MODELS)
def test_the_new_architectures_are_scoped_to_german(model_key: str) -> None:
    cfg = _load(CONFIG_DIR / f"{model_key}.yaml")
    assert cfg["cultures"] == ["german"]
    assert cfg["mlflow"]["experiment"] == "culture_mllm_training"


@pytest.mark.parametrize(
    ("model_key", "expected_id", "expected_modality"),
    [
        ("qwen3_vl_2b", "Qwen/Qwen3-VL-2B-Thinking", "vision_text"),
        ("llama3_2_3b", "meta-llama/Llama-3.2-3B", "text"),
    ],
)
def test_each_new_config_names_the_release_it_was_written_for(
    model_key: str, expected_id: str, expected_modality: str
) -> None:
    cfg = _load(CONFIG_DIR / f"{model_key}.yaml")
    assert cfg["model"]["id"] == expected_id
    assert resolve_modality(cfg["model"]) == expected_modality
    assert get_model_key(cfg, CONFIG_DIR / f"{model_key}.yaml") == model_key


@pytest.mark.parametrize("model_key", ["qwen3_vl_2b", "llama3_2_3b"])
def test_the_small_new_models_keep_the_shared_effective_batch_of_sixteen(
    model_key: str,
) -> None:
    train_cfg = _load(CONFIG_DIR / f"{model_key}.yaml")["training"]
    assert train_cfg["batch_size"] * train_cfg["gradient_accumulation"] == 16


@pytest.mark.parametrize("track", ["distributional", "subpop"])
def test_the_30b_first_token_configs_fit_the_card_the_way_the_wvs_config_does(
    track: str,
) -> None:
    train_cfg = _load(CONFIG_DIR / f"muse_glimmer_30b_{track}.yaml")["training"]
    base_cfg = _load(CONFIG_DIR / "muse_glimmer_30b.yaml")["training"]

    assert train_cfg["quantization"] == base_cfg["quantization"] == "4bit"
    assert train_cfg["gradient_checkpointing"] is True


@pytest.mark.parametrize("track", ["distributional", "subpop"])
def test_the_30b_first_token_configs_keep_the_recipe_of_the_2b_ones(track: str) -> None:
    muse = _load(CONFIG_DIR / f"muse_glimmer_30b_{track}.yaml")
    qwen = _load(CONFIG_DIR / f"qwen3_vl_2b_{track}.yaml")
    card_bound = {"batch_size", "gradient_accumulation", "gradient_checkpointing", "quantization"}
    per_backbone = {"eval_steps", "save_steps", "early_stopping_patience", "max_seq_len"}

    def recipe(cfg: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in cfg["training"].items() if k not in card_bound | per_backbone}

    def effective_batch(cfg: dict[str, Any]) -> int:
        return int(cfg["training"]["batch_size"] * cfg["training"]["gradient_accumulation"])

    def patience_in_steps(cfg: dict[str, Any]) -> int:
        return int(cfg["training"]["early_stopping_patience"] * cfg["training"]["eval_steps"])

    assert muse["lora"] == qwen["lora"]
    assert muse["data"] == qwen["data"]
    assert recipe(muse) == recipe(qwen)
    assert effective_batch(muse) == effective_batch(qwen)
    assert patience_in_steps(muse) == patience_in_steps(qwen)


def test_a_text_only_modality_selects_the_tokenizer_side_lora_defaults() -> None:
    assert default_exclude_modules("text") is None
    assert default_exclude_modules("vision_text") == ".*(vision_tower|audio_tower).*"
