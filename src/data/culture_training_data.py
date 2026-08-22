"""
CultureLLM → VLM adaptation: load per-culture WVS training data.

Reads the existing CultureLLM fine-tuning JSONL files (WVS text Q&A) and
normalises them to the standard HuggingFace chat-completion format used for
VLM SFT. These text-only examples anchor cultural values in the model's
language path before visual fine-tuning begins.

Usage:
    uv run python src/data/culture_training_data.py --culture arabic --dry-run
    uv run python src/data/culture_training_data.py --all
"""

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console

from src.data.cultures import (
    CULTURE_DIR_MAP,
    CULTURES,
    DERIVED_CULTURES,
    EXTRA_CULTURE_CONTEXTS,
    DerivedCultureSpec,
)

load_dotenv()
console = Console()

CULTURELLM_DATA_DIR = os.getenv("CULTURELLM_DATA_DIR", "../CultureLLM/data")
CULTURE_CONTEXT_JSONL = os.getenv(
    "CULTURE_CONTEXT_JSONL", "../CultureLLM/data/culture_context.jsonl"
)
OUTPUT_DIR = Path("data/processed")

NEUTRAL_SYSTEM_PROMPT = "You are a helpful assistant."

_CONTEXT_KEY_REMAP = {"germany": "german", "turkey": "turkish", "china": "chinese"}


def load_culture_contexts() -> dict[str, str]:
    """Load the per-culture prose blurbs, extended with this repo's own entries.

    Cultures derived here have no line in CultureLLM's ``culture_context.jsonl``,
    so ``EXTRA_CULTURE_CONTEXTS`` is merged on top.  The merge is not cosmetic:
    visual training looks contexts up with ``.get`` and silently falls back to the
    baseline system prompt on a miss, which would quietly turn the cultural
    condition into a second baseline.
    """
    path = Path(CULTURE_CONTEXT_JSONL)
    if not path.exists():
        raise FileNotFoundError(f"Culture context file not found: {path}")
    contexts: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        canonical = k.lower()
                        canonical = _CONTEXT_KEY_REMAP.get(canonical, canonical)
                        contexts[canonical] = v
    contexts.update(EXTRA_CULTURE_CONTEXTS)
    return contexts


def _wvs_prompt(question: dict[str, Any]) -> str:
    """Render one WVS question the way CultureLLM's ``getPrompt`` renders it.

    The scale bounds come from a lexicographic min/max over the digits found in the
    option string, and questions that do not already read as questions are wrapped
    in ``Do you agree with ...?``.  Both quirks are reproduced verbatim: a prompt
    that differs by a character no longer lines up with the ``spanish`` partition
    this culture is meant to be compared against.
    """
    content = question["q_content"]
    option = question["option"]
    nums = re.findall(r"\d+", option)
    body = f"{content} {option}" if "?" in content else f"Do you agree with {content}? {option}"
    return (
        f"Give me the answer from {min(nums)} to {max(nums)}: "
        f"{body}. You can only choose one option."
    )


def _aggregate_answers(csv_path: Path, aggregate_row: str) -> dict[str, int]:
    """Read one row of country-mean WVS answers, keyed by question ID.

    Means are truncated toward zero rather than rounded, and negative means are
    sign-flipped, because CultureLLM sums the ``-1``/``-2``/``-5`` "no answer"
    sentinels into the mean and then takes the absolute value.
    """
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            if row["B_COUNTRY"] != aggregate_row:
                continue
            answers = {}
            for key, value in row.items():
                if key.startswith("Q") and value:
                    answer = int(float(value))
                    answers[key[1:]] = -answer if answer < 0 else answer
            return answers
    raise ValueError(f"No {aggregate_row!r} row in {csv_path}")


def build_derived_culture_examples(spec: DerivedCultureSpec) -> list[dict[str, Any]]:
    """Assemble a culture CultureLLM never shipped a fine-tuning file for.

    Each question bank is paired with one country's mean answers, producing the
    same chat records upstream's ``generateFintuneData`` would have written had it
    been run for that country alone.
    """
    root = Path(CULTURELLM_DATA_DIR)
    answers = _aggregate_answers(root / spec.country_csv, spec.aggregate_row)
    token = spec.system_prompt_token
    system = f"You are an {token} chatbot that know {token} very well."

    examples = []
    for name in spec.question_files:
        with open(root / name) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                question = json.loads(line)
                examples.append(
                    {
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": _wvs_prompt(question)},
                            {"role": "assistant", "content": str(answers[question["q_id"]])},
                        ]
                    }
                )
    return examples


def load_wvs_culture_data(culture: str) -> list[dict[str, Any]]:
    """
    Load per-culture WVS Q&A from CultureLLM.

    The CultureLLM files already use chat-completion format. We normalise
    them so the 'content' field is always a plain string (not a list), making
    them compatible with VLM SFT trainers that process text-only turns.

    File selection is prioritised: the ``_1000.jsonl`` files hold the largest
    WVS set and are preferred, while the ``sentence_only`` and ``llama``
    variants are excluded.

    Derived cultures have no such file and are rebuilt from a country aggregate
    instead, so they take the ``DERIVED_CULTURES`` path.
    """
    spec = DERIVED_CULTURES.get(culture)
    if spec is not None:
        examples = build_derived_culture_examples(spec)
        console.print(
            f"  [green]{culture}[/green]: built {len(examples)} WVS examples "
            f"from {spec.country_csv} ({spec.aggregate_row})"
        )
        return examples

    dir_name = CULTURE_DIR_MAP.get(culture)
    if not dir_name:
        raise ValueError(f"Unknown culture: {culture}")

    finetune_dir = Path(CULTURELLM_DATA_DIR) / dir_name / "Finetune"
    if not finetune_dir.exists():
        console.print(f"[yellow]No Finetune dir for {culture}: {finetune_dir}[/yellow]")
        return []

    pattern = str(finetune_dir / f"WVQ_{dir_name}_*.jsonl")
    files = glob.glob(pattern)

    clean = [f for f in files if "_sentence_only" not in f and "_llama" not in f and "_L." not in f]
    preferred_1000 = [f for f in clean if f.endswith("_1000.jsonl")]
    files_to_use = preferred_1000 if preferred_1000 else clean if clean else files

    examples = []
    for fpath in files_to_use:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                msgs = obj.get("messages", [])
                normalised = []
                for msg in msgs:
                    content = msg["content"]
                    if isinstance(content, list):
                        content = " ".join(
                            item.get("text", "") if isinstance(item, dict) else str(item)
                            for item in content
                        )
                    normalised.append({"role": msg["role"], "content": content})
                examples.append({"messages": normalised})

    console.print(
        f"  [green]{culture}[/green]: loaded {len(examples)} WVS examples "
        f"from {len(files_to_use)} files"
    )
    return examples


def make_baseline_variant(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace the culture-specific system prompt with a neutral one."""
    out = []
    for ex in examples:
        msgs = list(ex["messages"])
        msgs[0] = {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT}
        out.append({"messages": msgs})
    return out


def save_culture_wvs_data(
    culture: str, examples: list[dict[str, Any]], condition: str = "cultural"
) -> Path:
    out_dir = OUTPUT_DIR / culture
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        "wvs_cultural_anchoring.jsonl"
        if condition == "cultural"
        else "wvs_baseline_anchoring.jsonl"
    )
    out_path = out_dir / filename
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Load CultureLLM WVS data for VLM training.")
    parser.add_argument("--culture", choices=CULTURES, help="Single culture to process")
    parser.add_argument("--all", action="store_true", help="Process all cultures")
    parser.add_argument("--dry-run", action="store_true", help="Print 5 samples, do not save")
    args = parser.parse_args()

    console.rule("[bold blue]CultureLLM WVS Data Loader[/bold blue]")

    contexts = load_culture_contexts()
    console.print(f"Loaded culture contexts for: {list(contexts.keys())}\n")

    cultures_to_process: list[str] = (
        list(CULTURES) if args.all else ([args.culture] if args.culture else [])
    )
    if not cultures_to_process:
        parser.error("Specify --culture <name> or --all")

    for culture in cultures_to_process:
        examples = load_wvs_culture_data(culture)
        if not examples:
            continue

        if args.dry_run:
            console.print(f"\n[bold]Sample records for {culture}:[/bold]")
            for ex in examples[:5]:
                console.print(json.dumps(ex, indent=2, ensure_ascii=False))
        else:
            out_path = save_culture_wvs_data(culture, examples, condition="cultural")
            console.print(f"  Saved {len(examples)} examples → {out_path}")
            baseline = make_baseline_variant(examples)
            out_path_b = save_culture_wvs_data(culture, baseline, condition="baseline")
            console.print(f"  Saved {len(baseline)} baseline examples → {out_path_b}")

    console.print("\n[bold green]✓ Done[/bold green]")


if __name__ == "__main__":
    main()
