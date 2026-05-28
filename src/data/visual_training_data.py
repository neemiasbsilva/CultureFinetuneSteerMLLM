"""
Build visual SFT training examples from PerceptSent images + σ₃P₅ labels.

For each (culture, fold) combination this module produces two JSONL files:
  - {culture}/fold_{n}_train.jsonl  — cultural condition (culture context in system prompt)
  - baseline/fold_{n}_train.jsonl   — baseline condition (neutral system prompt)

Caption/justification text is bootstrapped from the existing MLLM captions in
percept_dataset_alpha5_p5.csv. Tags are extracted as noun-phrase keywords
filtered against the 593-label perception vocabulary from mllm-persona-evaluation.

Usage:
    uv run python src/data/visual_training_data.py --culture arabic --fold 1
    uv run python src/data/visual_training_data.py --all-cultures --all-folds
    uv run python src/data/visual_training_data.py --culture arabic --fold 1 --dry-run
"""

import argparse
import json
import os
import re
from pathlib import Path

import nltk
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

AGREEMENT_CSV = os.getenv(
    "AGREEMENT_CSV",
    "../multimodal-LLMs-see-sentiment/data/agreement_p5_sigma3.csv",
)
EXISTING_CAPTIONS_CSV = os.getenv(
    "EXISTING_CAPTIONS_CSV",
    "../multimodal-LLMs-see-sentiment/reports/percept_dataset_alpha5_p5.csv",
)
CULTURE_CONTEXT_JSONL = os.getenv(
    "CULTURE_CONTEXT_JSONL", "../CultureLLM/data/culture_context.jsonl"
)
PERCEPTIONS_VOCAB_JSON = os.getenv(
    "PERCEPTIONS_VOCAB_JSON",
    "../mllm-persona-evaluation/data/unique_perceptions.json",
)
IMAGES_DIR = os.getenv("PERCEPTSENT_IMAGES_DIR", "../perceptsent/images")
FOLDS_DIR = Path("data/folds")
OUTPUT_DIR = Path("data/processed")

CULTURES = [
    "arabic", "bengali", "chinese", "english", "german",
    "korean", "portuguese", "spanish", "turkish",
]

SENTIMENT_LABELS = {0: "negative", 1: "slightly_negative", 2: "neutral",
                    3: "slightly_positive", 4: "positive"}

USER_PROMPT_JSON = (
    "Analyze this urban scene and provide your sentiment assessment. "
    "Respond only with this exact JSON structure:\n"
    '{"sentiment": <0-4>, "caption": "<one sentence>", '
    '"justification": "<2-3 sentences>", "tags": ["<tag1>", ...]}\n'
    "Sentiment scale: 0=negative, 1=slightly negative, 2=neutral, "
    "3=slightly positive, 4=positive. "
    "Tags should be concise descriptors of the urban scene."
)

SYSTEM_PROMPT_CULTURAL = (
    "You are an expert in urban visual sentiment analysis with deep roots in {culture} culture. "
    "{context}"
)

SYSTEM_PROMPT_BASELINE = (
    "You are an expert in urban visual sentiment analysis. "
    "Analyze urban scenes objectively and accurately."
)


def _download_nltk_data() -> None:
    for pkg_key, pkg_name in [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("corpora/stopwords", "stopwords"),
    ]:
        try:
            nltk.data.find(pkg_key)
        except LookupError:
            nltk.download(pkg_name, quiet=True)


def load_culture_contexts() -> dict[str, str]:
    from src.data.culture_training_data import load_culture_contexts as _load
    return _load()


def load_perception_vocab() -> set[str]:
    path = Path(PERCEPTIONS_VOCAB_JSON)
    if not path.exists():
        console.print(f"[yellow]Perception vocab not found at {path}, using empty set[/yellow]")
        return set()
    with open(path) as f:
        data = json.load(f)
    # The file has format {"unique_perceptions": [...]} or a plain list
    if isinstance(data, dict):
        vocab = data.get("unique_perceptions", list(data.values())[0] if data else [])
    else:
        vocab = data
    return {tag.lower() for tag in vocab}


def parse_caption_and_justification(caption_text: str) -> tuple[str, str]:
    """
    The existing captions have format: "Positive. The image shows..."
    Split into (caption, justification) heuristically.
    """
    caption_text = caption_text.strip()
    # Sentiment keyword prefix (e.g. "Positive. " or "Negative. ")
    caption_text = re.sub(r'^(Very\s+)?(Positive|Negative|Neutral|Slightly\s+\w+)\.\s*', '', caption_text, flags=re.IGNORECASE)
    # Split at sentence boundary: first sentence = caption, rest = justification
    sentences = re.split(r'(?<=[.!?])\s+', caption_text.strip())
    if len(sentences) == 1:
        return sentences[0], sentences[0]
    caption = sentences[0]
    justification = " ".join(sentences[1:])
    return caption, justification


def extract_tags(caption_text: str, vocab: set[str], max_tags: int = 5) -> list[str]:
    """Extract noun phrases from caption and match against perception vocabulary."""
    _download_nltk_data()
    tokens = nltk.word_tokenize(caption_text.lower())
    tags_from_vocab = [t for t in vocab if t in caption_text.lower()][:max_tags]
    if tags_from_vocab:
        return tags_from_vocab[:max_tags]
    # Fallback: extract nouns via POS tagging
    pos_tags = nltk.pos_tag(tokens)
    nouns = [word for word, pos in pos_tags if pos in ("NN", "NNS", "NNP", "NNPS")]
    stop = {"image", "scene", "area", "place", "building", "photo", "picture"}
    nouns = [n for n in nouns if n not in stop and len(n) > 3]
    return list(dict.fromkeys(nouns))[:max_tags]  # dedup, preserve order


def load_existing_captions() -> dict[str, dict]:
    """Returns {image_id: {caption, justification}} from existing MLLM outputs."""
    path = Path(EXISTING_CAPTIONS_CSV)
    if not path.exists():
        console.print(f"[yellow]Existing captions CSV not found: {path}[/yellow]")
        return {}
    df = pd.read_csv(path)
    result: dict[str, dict] = {}
    for _, row in df.iterrows():
        # image_path is like "/path/to/{id}.jpg"
        img_path = str(row.get("image_path", ""))
        image_id = Path(img_path).stem
        caption_raw = str(row.get("caption", ""))
        cap, just = parse_caption_and_justification(caption_raw)
        result[image_id] = {"caption": cap, "justification": just, "raw": caption_raw}
    return result


def build_example(
    image_id: str,
    image_path: str,
    sentiment: int,
    caption: str,
    justification: str,
    tags: list[str],
    culture: str | None,
    culture_context: str | None,
) -> dict:
    if culture and culture_context:
        system = SYSTEM_PROMPT_CULTURAL.format(culture=culture, context=culture_context)
        condition = "cultural"
    else:
        system = SYSTEM_PROMPT_BASELINE
        condition = "baseline"

    assistant_output = json.dumps({
        "sentiment": sentiment,
        "caption": caption,
        "justification": justification,
        "tags": tags,
    }, ensure_ascii=False)

    return {
        "condition": condition,
        "culture": culture or "baseline",
        "image_id": image_id,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": USER_PROMPT_JSON},
                ],
            },
            {"role": "assistant", "content": assistant_output},
        ],
    }


def generate_fold_data(
    culture: str | None,
    fold_n: int,
    split: str,
    contexts: dict[str, str],
    existing_captions: dict[str, dict],
    vocab: set[str],
) -> list[dict]:
    fold_path = FOLDS_DIR / f"fold_{fold_n}_{split}.csv"
    if not fold_path.exists():
        raise FileNotFoundError(
            f"Fold file not found: {fold_path}. Run make_folds.py first."
        )
    df = pd.read_csv(fold_path)
    images_path = Path(IMAGES_DIR)
    examples = []

    context = contexts.get(culture) if culture else None

    for _, row in df.iterrows():
        image_id = str(row["image_id"])
        image_path = str(images_path / f"{image_id}.jpg")
        sentiment = int(row["sentiment"])

        cap_data = existing_captions.get(image_id, {})
        caption = cap_data.get("caption", f"An urban scene with sentiment {SENTIMENT_LABELS[sentiment]}.")
        justification = cap_data.get("justification", "The visual elements of the scene suggest this emotional tone.")
        tags = extract_tags(cap_data.get("raw", caption), vocab)

        ex = build_example(image_id, image_path, sentiment, caption, justification, tags, culture, context)
        examples.append(ex)

    return examples


def save_examples(examples: list[dict], culture: str | None, fold_n: int, split: str) -> Path:
    cond_dir = OUTPUT_DIR / (culture if culture else "baseline")
    cond_dir.mkdir(parents=True, exist_ok=True)
    out_path = cond_dir / f"fold_{fold_n}_{split}.jsonl"
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build visual VLM SFT training examples.")
    parser.add_argument("--culture", choices=CULTURES + ["baseline"])
    parser.add_argument("--all-cultures", action="store_true")
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    console.rule("[bold blue]Visual Training Data Builder[/bold blue]")

    contexts = load_culture_contexts()
    existing_captions = load_existing_captions()
    vocab = load_perception_vocab()
    console.print(f"Loaded {len(existing_captions)} captions, {len(vocab)} perception tags\n")

    cultures_to_run: list[str | None] = []
    if args.all_cultures:
        cultures_to_run = CULTURES + [None]  # None = baseline
    elif args.culture:
        cultures_to_run = [None if args.culture == "baseline" else args.culture]
    else:
        parser.error("Specify --culture or --all-cultures")

    folds_to_run = list(range(1, 6)) if args.all_folds else ([args.fold] if args.fold else [])
    if not folds_to_run:
        parser.error("Specify --fold or --all-folds")

    for culture in cultures_to_run:
        for fold_n in folds_to_run:
            label = culture if culture else "baseline"
            for split in ("train", "val"):
                console.print(f"[bold]Building[/bold] {label} / fold {fold_n} / {split}...")
                examples = generate_fold_data(culture, fold_n, split, contexts, existing_captions, vocab)

                if args.dry_run:
                    if split == "train":
                        for ex in examples[:2]:
                            console.print(json.dumps(ex, indent=2, ensure_ascii=False)[:600])
                    console.print(f"  → would write {len(examples)} {split} examples")
                else:
                    out_path = save_examples(examples, culture, fold_n, split)
                    console.print(f"  → {len(examples)} examples → {out_path}")

    console.print("\n[bold green]✓ Done[/bold green]")


if __name__ == "__main__":
    main()
