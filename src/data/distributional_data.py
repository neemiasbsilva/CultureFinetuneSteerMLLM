"""Country-level WVS response distributions from Cao et al. (NAACL 2025), records to tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()
console = Console()

SOURCE_REPO_URL = "https://github.com/yongcaoplus/SimLLMCultureDist"
SOURCE_COMMIT = "402ee0b350983fd665e6668d770c7a687c5669a6"
RAW_DIR = Path(os.getenv("DISTRIBUTIONAL_RAW_DIR", "data/raw/distributional"))
PROVENANCE_FILENAME = "provenance.json"
PROVENANCE_PATH = RAW_DIR / PROVENANCE_FILENAME
SPLITS: tuple[str, ...] = ("train", "valid", "test")
SPLIT_FILES: dict[str, str] = {split: f"sft_wvs_{split}.json" for split in SPLITS}

PSEUDO_CULTURE = "global"
CONDITION = "distributional"
DATA_FORMAT = "distributional"

OPTION_LETTERS: tuple[str, ...] = tuple("ABCDEFGHIJX")
MAX_OPTIONS = len(OPTION_LETTERS)
ANSWER_PREFIX = " If had to select one of the options, my answer would be ("
TARGET_PAD = -1.0
PERCENT_TOLERANCE = 1e-3

EVALUATION_ITEMS: dict[str, str] = {
    "46": "happiness",
    "57": "social trust",
    "171": "religious attendance",
    "240": "left-right self-placement",
}

WVS_ATTRIBUTION = (
    "Haerpfer, C., Inglehart, R., Moreno, A., Welzel, C., Kizilova, K., Diez-Medrano, J., "
    "Lagos, M., Norris, P., Ponarin, E. & Puranen, B. (eds.). 2022. World Values Survey: "
    "Round Seven - Country-Pooled Datafile Version 5.0. Madrid, Spain & Vienna, Austria: "
    "JD Systems Institute & WVSA Secretariat. The response distributions used here are "
    "the per-country percentages Cao et al. derived from that file."
)
METHOD_ATTRIBUTION = (
    "Cao, Y., Liu, H., Arora, A., Augenstein, I., Röttger, P. & Hershcovich, D. 2025. "
    "Specializing Large Language Models to Simulate Survey Response Distributions for "
    "Global Populations. NAACL 2025 (arXiv:2502.07068). Data and loss definition from "
    f"{SOURCE_REPO_URL} at commit {SOURCE_COMMIT}; that repository ships no license file."
)

_COUNTRY_PATTERN = re.compile(r"^How would someone from (.+?) answer the following question:")
_OPTION_PATTERN = re.compile(r"^\(([A-Z])\)")


class RecordError(ValueError):
    pass


class OptionTokenError(ValueError):
    pass


@dataclass(frozen=True)
class SurveyRecord:
    id: str
    country: str
    instruction: str
    input: str
    options: tuple[str, ...]
    letters: tuple[str, ...]
    distribution: tuple[float, ...]
    data_type: str


def country_of(instruction: str) -> str:
    match = _COUNTRY_PATTERN.match(instruction)
    if match is None:
        raise RecordError(f"instruction does not name a country: {instruction!r}")
    return match.group(1)


def parse_record(raw: Mapping[str, Any]) -> SurveyRecord:
    record_id = str(raw["id"])
    options = tuple(str(option) for option in raw["options"])
    letters: list[str] = []
    for option in options:
        match = _OPTION_PATTERN.match(option)
        if match is None:
            raise RecordError(f"record {record_id!r}: option {option!r} lacks a (letter) prefix")
        letter = match.group(1)
        if letter not in OPTION_LETTERS:
            raise RecordError(
                f"record {record_id!r}: option letter {letter!r} is not one of "
                f"{''.join(OPTION_LETTERS)}"
            )
        if letter in letters:
            raise RecordError(f"record {record_id!r}: repeats option letter {letter!r}")
        letters.append(letter)

    percentages: Mapping[str, Any] = raw["options_dist"]
    if set(percentages) != set(letters):
        raise RecordError(
            f"record {record_id!r}: distribution keys {sorted(percentages)} do not match "
            f"the option letters {letters}"
        )
    values = [float(percentages[letter]) for letter in letters]
    for letter, value in zip(letters, values, strict=True):
        if value < 0:
            raise RecordError(f"record {record_id!r}: negative percentage for option {letter!r}")
    total = sum(values)
    if abs(total - 100.0) > PERCENT_TOLERANCE:
        raise RecordError(f"record {record_id!r}: percentages sum to {total:.4f}, not 100")

    return SurveyRecord(
        id=record_id,
        country=country_of(str(raw["instruction"])),
        instruction=str(raw["instruction"]),
        input=str(raw["input"]),
        options=options,
        letters=tuple(letters),
        distribution=tuple(value / 100.0 for value in values),
        data_type=str(raw["data_type"]),
    )


def load_split(split: str, raw_dir: Path = RAW_DIR) -> list[SurveyRecord]:
    if split not in SPLIT_FILES:
        raise ValueError(f"unknown split {split!r}; expected one of {', '.join(SPLITS)}")
    path = raw_dir / SPLIT_FILES[split]
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Run: uv run python src/data/distributional_data.py prepare "
            "--source-dir ../SimLLMCultureDist"
        )
    with open(path, encoding="utf-8") as handle:
        raw_records = json.load(handle)
    return [parse_record(raw) for raw in raw_records]


def count_items(
    records: Sequence[SurveyRecord], items: Mapping[str, str] = EVALUATION_ITEMS
) -> dict[str, int]:
    counts = dict.fromkeys(items, 0)
    for record in records:
        if record.id in counts:
            counts[record.id] += 1
    return counts


def filter_evaluation_items(
    records: Sequence[SurveyRecord], items: Mapping[str, str] = EVALUATION_ITEMS
) -> tuple[list[SurveyRecord], dict[str, int]]:
    kept = [record for record in records if record.id not in items]
    return kept, count_items(records, items)


def dedupe_records(records: Sequence[SurveyRecord]) -> list[SurveyRecord]:
    seen: set[tuple[str, str]] = set()
    unique: list[SurveyRecord] = []
    for record in records:
        key = (record.country, record.id)
        if key not in seen:
            seen.add(key)
            unique.append(record)
    return unique


def build_prompt(record: SurveyRecord) -> str:
    return record.instruction + record.input + ", ".join(record.options) + ANSWER_PREFIX


def _encode(tokenizer: Any, text: str) -> list[int]:
    return [int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"]]


def resolve_option_token_ids(
    tokenizer: Any,
    letters: Sequence[str] = OPTION_LETTERS,
    *,
    stub: str = ANSWER_PREFIX,
) -> dict[str, int]:
    stub_ids = _encode(tokenizer, stub)
    ids: dict[str, int] = {}
    for letter in letters:
        with_letter = _encode(tokenizer, stub + letter)
        if with_letter[: len(stub_ids)] != stub_ids:
            raise OptionTokenError(
                f"letter {letter!r}: the prefix tokens change when it follows {stub!r} "
                f"({stub_ids[-2:]} became {with_letter[len(stub_ids) - 2 : len(stub_ids)]}); "
                "the answer position would no longer be the token after the parenthesis"
            )
        tail = with_letter[len(stub_ids) :]
        if len(tail) != 1:
            raise OptionTokenError(
                f"letter {letter!r} is multi-token after {stub!r}: {tail}; the first-token "
                "objective needs exactly one token per option"
            )
        ids[letter] = tail[0]
    by_id: dict[int, list[str]] = {}
    for letter, token_id in ids.items():
        by_id.setdefault(token_id, []).append(letter)
    collisions = {token_id: group for token_id, group in by_id.items() if len(group) > 1}
    if collisions:
        raise OptionTokenError(f"option letters collide on token ids: {collisions}")
    return ids


def pad_token_id_for(tokenizer: Any) -> int:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None:
        return int(pad)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        return int(eos)
    raise ValueError("tokenizer has neither a pad nor an eos token; prompts cannot be padded")


@dataclass(frozen=True)
class PromptEncoder:
    tokenizer: Any
    letter_ids: Mapping[str, int]
    answer_token_id: int
    max_length: int

    @classmethod
    def build(cls, tokenizer: Any, *, max_length: int) -> PromptEncoder:
        letter_ids = resolve_option_token_ids(tokenizer)
        answer_token_id = _encode(tokenizer, ANSWER_PREFIX)[-1]
        return cls(tokenizer, letter_ids, answer_token_id, max_length)

    def encode(self, record: SurveyRecord) -> dict[str, list[int] | list[float]]:
        prompt = build_prompt(record)
        input_ids = [
            int(token) for token in self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        ]
        if len(input_ids) > self.max_length:
            raise RecordError(
                f"record {record.id!r} ({record.country}) is {len(input_ids)} tokens, above "
                f"max_length {self.max_length}; raise training.max_seq_len instead of truncating"
            )
        if input_ids[-1] != self.answer_token_id:
            raise RecordError(
                f"record {record.id!r}: the encoded prompt does not end with the answer "
                f"prefix token {self.answer_token_id}"
            )
        return {
            "input_ids": input_ids,
            "option_token_ids": [self.letter_ids[letter] for letter in record.letters],
            "target_dist": list(record.distribution),
        }

    def dataset(self, records: Sequence[SurveyRecord]) -> Dataset:
        return Dataset.from_list([self.encode(record) for record in records])


class DistributionalCollator:
    def __init__(self, pad_token_id: int, max_options: int = MAX_OPTIONS) -> None:
        self.pad_token_id = pad_token_id
        self.max_options = max_options

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        lengths = [len(feature["input_ids"]) for feature in features]
        width = max(lengths)
        rows = len(features)
        input_ids = torch.full((rows, width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((rows, width), dtype=torch.long)
        option_token_ids = torch.zeros((rows, self.max_options), dtype=torch.long)
        target_dist = torch.full((rows, self.max_options), TARGET_PAD, dtype=torch.float32)
        for row, feature in enumerate(features):
            tokens = torch.as_tensor(list(feature["input_ids"]), dtype=torch.long)
            input_ids[row, : len(tokens)] = tokens
            attention_mask[row, : len(tokens)] = 1
            option_ids = list(feature["option_token_ids"])
            if len(option_ids) > self.max_options:
                raise ValueError(
                    f"a record carries {len(option_ids)} options, more than the "
                    f"{self.max_options} the collator is sized for"
                )
            option_token_ids[row, : len(option_ids)] = torch.as_tensor(option_ids, dtype=torch.long)
            target_dist[row, : len(option_ids)] = torch.as_tensor(
                list(feature["target_dist"]), dtype=torch.float32
            )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answer_index": torch.as_tensor(lengths, dtype=torch.long) - 1,
            "option_token_ids": option_token_ids,
            "target_dist": target_dist,
        }


def split_summary(records: Sequence[SurveyRecord]) -> dict[str, int]:
    pairs = [(record.country, record.id) for record in records]
    return {
        "rows": len(records),
        "countries": len({record.country for record in records}),
        "questions": len({record.id for record in records}),
        "duplicate_pairs": len(pairs) - len(set(pairs)),
    }


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def copy_source_data(
    source_dir: Path,
    raw_dir: Path = RAW_DIR,
    *,
    expected_commit: str = SOURCE_COMMIT,
    revision: Callable[[Path], str] = git_head,
) -> dict[str, Any]:
    observed = revision(source_dir)
    if observed != expected_commit:
        raise ValueError(
            f"{source_dir} is at commit {observed}; this track pins {expected_commit}. "
            "Check out the pinned commit, or update SOURCE_COMMIT deliberately after "
            "re-verifying the data facts."
        )
    raw_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    files: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        name = SPLIT_FILES[split]
        source = source_dir / "dataset" / name
        if not source.is_file():
            raise FileNotFoundError(f"{source} is missing from the reference clone")
        digest = sha256_of(source)
        target = raw_dir / name
        if not target.is_file() or sha256_of(target) != digest:
            shutil.copyfile(source, target)
            copied += 1
        records = load_split(split, raw_dir)
        files[name] = {
            "split": split,
            "sha256": digest,
            "bytes": source.stat().st_size,
            **split_summary(records),
            "evaluation_item_rows": count_items(records),
        }
    return {
        "repo": SOURCE_REPO_URL,
        "commit": observed,
        "source_dir": str(source_dir.resolve()),
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "copied_files": copied,
        "files": files,
        "evaluation_items": dict(EVALUATION_ITEMS),
        "attribution": {"data": WVS_ATTRIBUTION, "method": METHOD_ATTRIBUTION},
    }


def write_provenance(raw_dir: Path, source: Mapping[str, Any]) -> Path:
    path = raw_dir / PROVENANCE_FILENAME
    partial = path.with_name(path.name + ".part")
    partial.write_text(
        json.dumps(source, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(partial, path)
    return path


def _print_stats(raw_dir: Path) -> None:
    table = Table(title=f"Reference splits under {raw_dir}")
    table.add_column("split")
    for column in ("rows", "countries", "questions", "duplicate_pairs"):
        table.add_column(column, justify="right")
    for item_id, label in EVALUATION_ITEMS.items():
        table.add_column(f"Q{item_id} {label}", justify="right")
    for split in SPLITS:
        records = load_split(split, raw_dir)
        summary = split_summary(records)
        items = count_items(records)
        table.add_row(
            split,
            *(
                str(summary[column])
                for column in ("rows", "countries", "questions", "duplicate_pairs")
            ),
            *(str(items[item_id]) for item_id in EVALUATION_ITEMS),
        )
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy and describe the Cao et al. WVS response-distribution tables."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="copy the splits and write provenance.json")
    prepare.add_argument("--source-dir", type=Path, default=Path("../SimLLMCultureDist"))
    prepare.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    stats = subparsers.add_parser("stats", help="summarize the copied splits")
    stats.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    if args.command == "prepare":
        provenance = copy_source_data(args.source_dir, args.raw_dir)
        path = write_provenance(args.raw_dir, provenance)
        console.print(
            f"Copied {provenance['copied_files']} file(s) from {args.source_dir} at commit "
            f"{provenance['commit'][:12]}; provenance at {path}"
        )
    _print_stats(args.raw_dir)


if __name__ == "__main__":
    main()
