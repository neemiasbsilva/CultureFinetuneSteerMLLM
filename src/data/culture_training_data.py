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
import glob
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

CULTURELLM_DATA_DIR = os.getenv("CULTURELLM_DATA_DIR", "../CultureLLM/data")
CULTURE_CONTEXT_JSONL = os.getenv(
    "CULTURE_CONTEXT_JSONL", "../CultureLLM/data/culture_context.jsonl"
)
OUTPUT_DIR = Path("data/processed")

CULTURES = [
    "arabic", "bengali", "chinese", "english", "german",
    "korean", "portuguese", "spanish", "turkish",
]

NEUTRAL_SYSTEM_PROMPT = "You are a helpful assistant."

CULTURE_DIR_MAP = {
    "arabic": "Arabic",
    "bengali": "Bengali",
    "chinese": "China",
    "english": "English",
    "german": "Germany",
    "korean": "Korean",
    "portuguese": "Portuguese",
    "spanish": "Spanish",
    "turkish": "Turkey",
}


_CONTEXT_KEY_REMAP = {"germany": "german", "turkey": "turkish", "china": "chinese"}


def load_culture_contexts() -> dict[str, str]:
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
    return contexts


def load_wvs_culture_data(culture: str) -> list[dict]:
    """
    Load per-culture WVS Q&A from CultureLLM.

    The CultureLLM files already use chat-completion format. We normalise
    them so the 'content' field is always a plain string (not a list), making
    them compatible with VLM SFT trainers that process text-only turns.
    """
    dir_name = CULTURE_DIR_MAP.get(culture)
    if not dir_name:
        raise ValueError(f"Unknown culture: {culture}")

    finetune_dir = Path(CULTURELLM_DATA_DIR) / dir_name / "Finetune"
    if not finetune_dir.exists():
        console.print(f"[yellow]No Finetune dir for {culture}: {finetune_dir}[/yellow]")
        return []

    pattern = str(finetune_dir / f"WVQ_{dir_name}_*.jsonl")
    files = glob.glob(pattern)

    # Priority: _1000.jsonl (largest WVS set) → exclude sentence_only/llama variants
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
                        # Flatten list content to string for text-only turns
                        content = " ".join(
                            item.get("text", "") if isinstance(item, dict) else str(item)
                            for item in content
                        )
                    normalised.append({"role": msg["role"], "content": content})
                examples.append({"messages": normalised})

    console.print(f"  [green]{culture}[/green]: loaded {len(examples)} WVS examples from {len(files_to_use)} files")
    return examples


def make_baseline_variant(examples: list[dict]) -> list[dict]:
    """Replace the culture-specific system prompt with a neutral one."""
    out = []
    for ex in examples:
        msgs = list(ex["messages"])
        msgs[0] = {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT}
        out.append({"messages": msgs})
    return out


def save_culture_wvs_data(culture: str, examples: list[dict], condition: str = "cultural") -> Path:
    out_dir = OUTPUT_DIR / culture
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = "wvs_cultural_anchoring.jsonl" if condition == "cultural" else "wvs_baseline_anchoring.jsonl"
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

    cultures_to_process = CULTURES if args.all else ([args.culture] if args.culture else [])
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
