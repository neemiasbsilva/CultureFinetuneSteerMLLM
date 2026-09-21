"""Shared fixtures and collection rules for the test suite."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
import pytest

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


class FakeSurveyTokenizer:
    def __init__(
        self,
        *,
        bos: bool = False,
        merge_paren: bool = False,
        split_letter: str | None = None,
        collide_letters: bool = False,
        pad_token_id: int | None = 0,
        eos_token_id: int | None = 1,
    ) -> None:
        self.bos = bos
        self.merge_paren = merge_paren
        self.split_letter = split_letter
        self.collide_letters = collide_letters
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.vocab: dict[str, int] = {}

    def _pieces(self, text: str) -> list[str]:
        pattern = r"\([A-Z]|\(|\w+|[^\w\s]" if self.merge_paren else r"\(|\w+|[^\w\s]"
        pieces: list[str] = []
        for piece in re.findall(pattern, text):
            if piece == self.split_letter:
                pieces.extend([piece + "#1", piece + "#2"])
            else:
                pieces.append(piece)
        return pieces

    def _id(self, piece: str) -> int:
        key = "LETTER" if self.collide_letters and re.fullmatch(r"[A-Z]", piece) else piece
        return self.vocab.setdefault(key, len(self.vocab) + 10)

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        ids = [self._id(piece) for piece in self._pieces(text)]
        if self.bos and add_special_tokens:
            ids = [2, *ids]
        return {"input_ids": ids}


@pytest.fixture
def survey_tokenizer() -> Callable[..., FakeSurveyTokenizer]:
    return FakeSurveyTokenizer


def _raw_survey_record(
    record_id: str = "46",
    country: str = "Andorra",
    options: tuple[str, ...] = ("(A)Very happy", "(B)Quite happy", "(C)Not very happy"),
    dist: dict[str, float] | None = None,
    data_type: str = "train",
) -> dict[str, Any]:
    return {
        "id": record_id,
        "instruction": f"How would someone from {country} answer the following question:\n\n",
        "input": "Taking all things together, would you say you are? \nHere are the options: \n",
        "options": list(options),
        "options_dist": dict(dist) if dist is not None else {"B": 60.0, "A": 30.0, "C": 10.0},
        "data_type": data_type,
    }


@pytest.fixture
def raw_survey_record() -> Callable[..., dict[str, Any]]:
    return _raw_survey_record


SUBPOP_STEERING_ENTRIES: list[dict[str, str]] = [
    {
        "attribute": "POLIDEOLOGY",
        "qa_prompt": "In general, would you describe your political views as",
        "options": "['Very conservative', 'Conservative', 'Moderate', 'Liberal', 'Very liberal']",
    },
    {
        "attribute": "SEX",
        "qa_prompt": "What is the sex that you were assigned at birth?",
        "options": "['Male', 'Female']",
    },
]


def _raw_subpop_record(
    qkey: str = "REASONGUND_W26",
    attribute: str = "POLIDEOLOGY",
    group: str = "Liberal",
    question: str = (
        "Please indicate whether the following is a major reason, a minor reason, or not "
        "a reason why you own a gun. As part of a gun collection"
    ),
    options: tuple[str, ...] = ("Major reason", "Minor reason", "Not a reason", "Refused"),
    responses: tuple[float, ...] = (0.1, 0.3, 0.6),
    refusal_rate: float = 0.02,
    ordinal: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    return {
        "qkey": qkey,
        "attribute": attribute,
        "group": group,
        "question": question,
        "options": list(options),
        "responses": list(responses),
        "refusal_rate": refusal_rate,
        "ordinal": list(ordinal) if ordinal is not None else [1.0] * len(responses),
    }


@pytest.fixture
def raw_subpop_record() -> Callable[..., dict[str, Any]]:
    return _raw_subpop_record


@pytest.fixture
def subpop_steering_entries() -> list[dict[str, str]]:
    return [dict(entry) for entry in SUBPOP_STEERING_ENTRIES]


class FakeLogitsModel:
    def __init__(self, logits: Any, *, supports_logits_to_keep: bool = True) -> None:
        self.logits = logits
        self.supports_logits_to_keep = supports_logits_to_keep
        self.device = "cpu"
        self.training = True
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        if "logits_to_keep" in kwargs and not self.supports_logits_to_keep:
            raise TypeError("forward() got an unexpected keyword argument 'logits_to_keep'")
        keep = kwargs.get("logits_to_keep", 0)
        if isinstance(keep, int):
            sliced = self.logits if keep == 0 else self.logits[:, -keep:, :]
        else:
            sliced = self.logits[:, keep, :]
        return SimpleNamespace(logits=sliced)

    def eval(self) -> None:
        self.training = False

    def train(self, mode: bool = True) -> None:
        self.training = mode


@pytest.fixture
def logits_model() -> Callable[..., FakeLogitsModel]:
    return FakeLogitsModel
