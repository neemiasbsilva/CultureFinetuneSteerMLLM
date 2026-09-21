"""Canonical annotation conditions and checkpoint layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    trained: bool
    checkpoint_subdir: str | None


CONDITION_REGISTRY: dict[str, ConditionSpec] = {
    "inference_only": ConditionSpec(
        name="inference_only",
        trained=False,
        checkpoint_subdir=None,
    ),
    "wvs_cultural": ConditionSpec(
        name="wvs_cultural",
        trained=True,
        checkpoint_subdir="cultural",
    ),
}

LEGACY_CONDITION_ALIASES: dict[str, str] = {
    "cultural": "wvs_cultural",
}

CONDITIONS: tuple[str, ...] = tuple(CONDITION_REGISTRY)
TRAINED_CONDITIONS: tuple[str, ...] = tuple(
    name for name, spec in CONDITION_REGISTRY.items() if spec.trained
)


def normalize_condition(condition: str, *, strict: bool = True) -> str:
    value = str(condition).strip().lower()
    value = LEGACY_CONDITION_ALIASES.get(value, value)
    if strict and value not in CONDITION_REGISTRY:
        allowed = ", ".join(CONDITIONS)
        raise ValueError(f"Unknown annotation condition {condition!r}; expected one of: {allowed}")
    return value


def get_condition_spec(condition: str) -> ConditionSpec:
    return CONDITION_REGISTRY[normalize_condition(condition)]


def checkpoint_directory(
    checkpoints_dir: str | Path,
    culture: str,
    model_name: str,
    condition: str,
) -> Path | None:
    spec = get_condition_spec(condition)
    if not spec.trained:
        return None
    assert spec.checkpoint_subdir is not None
    return Path(checkpoints_dir) / str(culture) / str(model_name) / spec.checkpoint_subdir
