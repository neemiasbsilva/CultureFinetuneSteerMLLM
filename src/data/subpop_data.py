"""US subpopulation response distributions from SubPOP (Suh et al. 2025), records to tensors."""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.table import Table

from src.data.distributional_data import (
    RecordError,
    git_head,
    resolve_option_token_ids,
    sha256_of,
    write_provenance,
)

load_dotenv()
console = Console()

DATASET_REPO_ID = "jjssuh/subpop"
DATASET_REVISION = "886d78a5cb55a576bd82b6724c46ae0f34a2ac3e"
DATASET_LICENSE = "cc-by-nc-sa-4.0"
DATASET_FILES: dict[str, str] = {"train": "subpop_train.jsonl", "eval": "subpop_eval.jsonl"}
SPLITS: tuple[str, ...] = tuple(DATASET_FILES)
SOURCE_REPO_URL = "https://github.com/JosephJeesungSuh/subpop"
SOURCE_COMMIT = "e0569aca48361415779aac7407aed1e2cbb67b4a"
STEERING_SOURCE = Path("data/subpopulation_metadata/steering_prompts.json")
STEERING_FILENAME = "steering_prompts.json"
RAW_DIR = Path(os.getenv("SUBPOP_RAW_DIR", "data/raw/subpop"))

PSEUDO_CULTURE = "subpop"
CONDITION = "subpop"
DATA_FORMAT = "subpop"
STEERING_TYPE = "qa"

OPTION_LETTERS: tuple[str, ...] = tuple(f" {letter}" for letter in "ABCDEFGHIJ")
MAX_OPTIONS = len(OPTION_LETTERS)
ANSWER_STUB = "Answer:"
STEERING_BRIDGE = "Answer the following question keeping in mind your previous answers.\n"
VAL_RATIO = 0.1
SPLIT_SEED = 42
UNIT_TOLERANCE = 1e-6

RELATED_ITEM_PATTERNS: dict[str, str] = {
    "happiness": r"\b(very|pretty|quite|rather) happy\b",
    "social trust": r"most people can be trusted|can.t be too careful",
    "religious attendance": r"attend religious services",
    "left-right self-placement": r"describe your (own )?political views",
}

DATA_ATTRIBUTION = (
    "SubPOP-Train is derived from Pew Research Center American Trends Panel waves and "
    "SubPOP-Eval from the NORC General Social Survey 2022. Pew Research Center and NORC "
    "bear no responsibility for the analyses or interpretations presented here; the "
    "opinions expressed are those of the authors. The tables are distributed through the "
    f"gated dataset {DATASET_REPO_ID} under {DATASET_LICENSE} for non-commercial research "
    "use and are never redistributed by this repository."
)
METHOD_ATTRIBUTION = (
    "Suh, J., Jahanparast, E., Moon, S., Kang, M. & Chang, S. 2025. Language Model "
    "Fine-Tuning on Scaled Survey Data for Predicting Distributions of Public Opinions "
    "(arXiv:2502.16761). Steering prompts, question format and loss definition from "
    f"{SOURCE_REPO_URL} at commit {SOURCE_COMMIT} (BSD 3-Clause)."
)

Downloader = Callable[..., str]


@dataclass(frozen=True)
class SubpopRecord:
    qkey: str
    attribute: str
    group: str
    question: str
    options: tuple[str, ...]
    responses: tuple[float, ...]
    refusal_rate: float
    ordinal: tuple[float, ...]

    @property
    def target(self) -> tuple[float, ...]:
        return (*self.responses, self.refusal_rate)


@dataclass(frozen=True)
class SteeringPrompt:
    attribute: str
    question: str
    options: tuple[str, ...]


def parse_record(raw: Mapping[str, Any]) -> SubpopRecord:
    qkey = str(raw["qkey"])
    options = tuple(str(option) for option in raw["options"])
    responses = tuple(float(value) for value in raw["responses"])
    ordinal = tuple(float(value) for value in raw["ordinal"])
    refusal_rate = float(raw["refusal_rate"])

    if len(options) > MAX_OPTIONS:
        raise RecordError(
            f"record {qkey!r}: {len(options)} options, more than the {MAX_OPTIONS} letters"
        )
    if len(options) != len(responses) + 1:
        raise RecordError(
            f"record {qkey!r}: {len(options)} options for {len(responses)} responses; the "
            "last option must be the refusal option the responses leave out"
        )
    if len(ordinal) != len(responses):
        raise RecordError(
            f"record {qkey!r}: {len(ordinal)} ordinal values for {len(responses)} responses"
        )
    if any(value < 0 for value in responses):
        raise RecordError(f"record {qkey!r}: negative response share")
    total = sum(responses)
    if abs(total - 1.0) > UNIT_TOLERANCE:
        raise RecordError(f"record {qkey!r}: responses sum to {total:.6f}, not 1")
    if not 0.0 <= refusal_rate <= 1.0:
        raise RecordError(f"record {qkey!r}: refusal rate {refusal_rate} is outside [0, 1]")

    return SubpopRecord(
        qkey=qkey,
        attribute=str(raw["attribute"]),
        group=str(raw["group"]),
        question=str(raw["question"]),
        options=options,
        responses=responses,
        refusal_rate=refusal_rate,
        ordinal=ordinal,
    )


def load_records(split: str, raw_dir: Path = RAW_DIR) -> list[SubpopRecord]:
    if split not in DATASET_FILES:
        raise ValueError(f"unknown split {split!r}; expected one of {', '.join(SPLITS)}")
    path = raw_dir / DATASET_FILES[split]
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Run: uv run python src/data/subpop_data.py prepare "
            "--source-dir ../subpop"
        )
    with open(path, encoding="utf-8") as handle:
        return [parse_record(json.loads(line)) for line in handle if line.strip()]


def load_steering(raw_dir: Path = RAW_DIR) -> dict[str, SteeringPrompt]:
    path = raw_dir / STEERING_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Run: uv run python src/data/subpop_data.py prepare "
            "--source-dir ../subpop"
        )
    with open(path, encoding="utf-8") as handle:
        entries = json.load(handle)
    steering: dict[str, SteeringPrompt] = {}
    for entry in entries:
        options = entry["options"]
        if isinstance(options, str):
            options = ast.literal_eval(options)
        attribute = str(entry["attribute"])
        steering[attribute] = SteeringPrompt(
            attribute=attribute,
            question=str(entry["qa_prompt"]),
            options=tuple(str(option) for option in options),
        )
    return steering


def format_mcq(body: str, options: Sequence[str], *, answer_forcing: bool = False) -> str:
    letters = [chr(ord("A") + index) for index in range(len(options))]
    lines = "\n".join(
        f"{letter}. {option.strip()}" for letter, option in zip(letters, options, strict=True)
    ).strip()
    forcing = (
        "Answer as a choice between " + ",".join(f"{letter}." for letter in letters)
        if answer_forcing
        else ""
    )
    question = f"Question: {body.strip()}\n{lines}\n{forcing}".strip()
    return f"{question}\n\n{ANSWER_STUB}"


def build_prompt(record: SubpopRecord, steering: Mapping[str, SteeringPrompt]) -> str:
    prompt = steering.get(record.attribute)
    if prompt is None:
        raise RecordError(f"record {record.qkey!r}: no steering prompt for {record.attribute!r}")
    if record.group not in prompt.options:
        raise RecordError(
            f"record {record.qkey!r}: group {record.group!r} is not a steering option of "
            f"{record.attribute!r}"
        )
    letter = chr(ord("A") + prompt.options.index(record.group))
    return (
        format_mcq(prompt.question, prompt.options)
        + f" {letter}. {record.group}\n\n"
        + STEERING_BRIDGE
        + format_mcq(record.question, record.options, answer_forcing=True)
    )


def split_by_question(
    records: Sequence[SubpopRecord], *, val_ratio: float = VAL_RATIO, seed: int = SPLIT_SEED
) -> tuple[list[SubpopRecord], list[SubpopRecord]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    qkeys = sorted({record.qkey for record in records})
    random.Random(seed).shuffle(qkeys)
    held_out = set(qkeys[: round(val_ratio * len(qkeys))])
    train = [record for record in records if record.qkey not in held_out]
    valid = [record for record in records if record.qkey in held_out]
    return train, valid


def related_qkeys(
    records: Sequence[SubpopRecord], patterns: Mapping[str, str] = RELATED_ITEM_PATTERNS
) -> dict[str, list[str]]:
    texts = {record.qkey: " ".join((record.question, *record.options)) for record in records}
    return {
        label: sorted(
            qkey for qkey, text in texts.items() if re.search(pattern, text, flags=re.IGNORECASE)
        )
        for label, pattern in patterns.items()
    }


def filter_related_items(
    records: Sequence[SubpopRecord], patterns: Mapping[str, str] = RELATED_ITEM_PATTERNS
) -> tuple[list[SubpopRecord], int]:
    related = {qkey for qkeys in related_qkeys(records, patterns).values() for qkey in qkeys}
    kept = [record for record in records if record.qkey not in related]
    return kept, len(records) - len(kept)


@dataclass(frozen=True)
class SubpopEncoder:
    tokenizer: Any
    steering: Mapping[str, SteeringPrompt]
    letter_ids: Mapping[str, int]
    answer_token_id: int
    max_length: int

    @classmethod
    def build(
        cls, tokenizer: Any, steering: Mapping[str, SteeringPrompt], *, max_length: int
    ) -> SubpopEncoder:
        letter_ids = resolve_option_token_ids(tokenizer, OPTION_LETTERS, stub=ANSWER_STUB)
        stub_ids = tokenizer(ANSWER_STUB, add_special_tokens=False)["input_ids"]
        return cls(tokenizer, steering, letter_ids, int(stub_ids[-1]), max_length)

    def encode(self, record: SubpopRecord) -> dict[str, list[int] | list[float]]:
        prompt = build_prompt(record, self.steering)
        input_ids = [
            int(token) for token in self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        ]
        if len(input_ids) > self.max_length:
            raise RecordError(
                f"record {record.qkey!r} ({record.attribute}/{record.group}) is "
                f"{len(input_ids)} tokens, above max_length {self.max_length}; raise "
                "training.max_seq_len instead of truncating"
            )
        if input_ids[-1] != self.answer_token_id:
            raise RecordError(
                f"record {record.qkey!r}: the encoded prompt does not end with the answer "
                f"stub token {self.answer_token_id}"
            )
        target = list(record.target)
        return {
            "input_ids": input_ids,
            "option_token_ids": [self.letter_ids[letter] for letter in OPTION_LETTERS],
            "target_dist": target + [0.0] * (MAX_OPTIONS - len(target)),
        }

    def dataset(self, records: Sequence[SubpopRecord]) -> Dataset:
        return Dataset.from_list([self.encode(record) for record in records])


def split_summary(records: Sequence[SubpopRecord]) -> dict[str, int]:
    return {
        "rows": len(records),
        "questions": len({record.qkey for record in records}),
        "groups": len({(record.attribute, record.group) for record in records}),
        "attributes": len({record.attribute for record in records}),
    }


def _copy_if_changed(source: Path, target: Path) -> int:
    if target.is_file() and sha256_of(target) == sha256_of(source):
        return 0
    shutil.copyfile(source, target)
    return 1


def fetch_source_data(
    source_dir: Path,
    raw_dir: Path = RAW_DIR,
    *,
    expected_commit: str = SOURCE_COMMIT,
    revision: Callable[[Path], str] = git_head,
    download: Downloader = hf_hub_download,
) -> dict[str, Any]:
    observed = revision(source_dir)
    if observed != expected_commit:
        raise ValueError(
            f"{source_dir} is at commit {observed}; this track pins {expected_commit}. "
            "Check out the pinned commit, or update SOURCE_COMMIT deliberately after "
            "re-verifying the steering prompts."
        )
    steering_source = source_dir / STEERING_SOURCE
    if not steering_source.is_file():
        raise FileNotFoundError(f"{steering_source} is missing from the reference clone")

    raw_dir.mkdir(parents=True, exist_ok=True)
    copied = _copy_if_changed(steering_source, raw_dir / STEERING_FILENAME)
    files: dict[str, dict[str, Any]] = {}
    for split, name in DATASET_FILES.items():
        cached = Path(
            download(
                repo_id=DATASET_REPO_ID,
                filename=name,
                repo_type="dataset",
                revision=DATASET_REVISION,
            )
        )
        copied += _copy_if_changed(cached, raw_dir / name)
        records = load_records(split, raw_dir)
        files[name] = {
            "split": split,
            "sha256": sha256_of(raw_dir / name),
            "bytes": (raw_dir / name).stat().st_size,
            **split_summary(records),
            "related_questions": related_qkeys(records),
        }
    return {
        "dataset": {
            "repo": DATASET_REPO_ID,
            "revision": DATASET_REVISION,
            "license": DATASET_LICENSE,
        },
        "code": {
            "repo": SOURCE_REPO_URL,
            "commit": observed,
            "source_dir": str(source_dir.resolve()),
            "steering_sha256": sha256_of(raw_dir / STEERING_FILENAME),
        },
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "copied_files": copied,
        "files": files,
        "split": {"by": "question", "val_ratio": VAL_RATIO, "seed": SPLIT_SEED},
        "related_item_patterns": dict(RELATED_ITEM_PATTERNS),
        "attribution": {"data": DATA_ATTRIBUTION, "method": METHOD_ATTRIBUTION},
    }


def _print_stats(raw_dir: Path) -> None:
    table = Table(title=f"SubPOP tables under {raw_dir}")
    table.add_column("split")
    for column in ("rows", "questions", "groups", "attributes"):
        table.add_column(column, justify="right")
    for label in RELATED_ITEM_PATTERNS:
        table.add_column(label, justify="right")
    for split in SPLITS:
        records = load_records(split, raw_dir)
        summary = split_summary(records)
        related = related_qkeys(records)
        table.add_row(
            split,
            *(str(summary[column]) for column in ("rows", "questions", "groups", "attributes")),
            *(str(len(related[label])) for label in RELATED_ITEM_PATTERNS),
        )
    console.print(table)
    train, valid = split_by_question(load_records("train", raw_dir))
    console.print(
        f"By-question split (seed {SPLIT_SEED}, val_ratio {VAL_RATIO}): "
        f"{len(train)} train rows / {len(valid)} valid rows"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and describe the SubPOP subpopulation response-distribution tables."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="fetch the tables and write provenance.json")
    prepare.add_argument("--source-dir", type=Path, default=Path("../subpop"))
    prepare.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    stats = subparsers.add_parser("stats", help="summarize the fetched tables")
    stats.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    if args.command == "prepare":
        provenance = fetch_source_data(args.source_dir, args.raw_dir)
        path = write_provenance(args.raw_dir, provenance)
        console.print(
            f"Copied {provenance['copied_files']} file(s) from {DATASET_REPO_ID} at revision "
            f"{DATASET_REVISION[:12]} and {args.source_dir} at commit "
            f"{provenance['code']['commit'][:12]}; provenance at {path}"
        )
    _print_stats(args.raw_dir)


if __name__ == "__main__":
    main()
