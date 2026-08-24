"""Push completed LoRA checkpoints to the Hugging Face Hub collection repo.

Usage:
    uv run python src/hub/push.py status
    uv run python src/hub/push.py push --dry-run
    uv run python src/hub/push.py push --cultures english german --models qwen3_5_2b
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import RepositoryNotFoundError

from src.data.cultures import CULTURES

DEFAULT_REPO_ID = "Neemias/Culture-Steering-MLLM-Collection"
DEFAULT_MODELS: tuple[str, ...] = (
    "gemma4_e2b",
    "gemma4_e4b",
    "gemma4_31b",
    "qwen3_5_2b",
    "qwen3_27b",
    "qwen3_vl_8b",
)
DEFAULT_PROBLEM = "cultural"
CONFIG_DIR = Path("configs")
CHECKPOINTS_DIR = Path("checkpoints")

PUSH_FILENAMES: tuple[str, ...] = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


@dataclass(frozen=True)
class Checkpoint:
    culture: str
    backbone: str
    problem: str
    path: Path
    base_model_id: str
    lora: dict[str, Any]

    @property
    def path_in_repo(self) -> str:
        return f"{self.culture}/{self.backbone}/{self.problem}"


def _load_config(config_dir: Path, backbone: str, problem: str) -> dict[str, Any]:
    yfcc_config = config_dir / f"{backbone}_yfcc.yaml"
    use_yfcc = problem.startswith("yfcc") and yfcc_config.is_file()
    config_path = yfcc_config if use_yfcc else config_dir / f"{backbone}.yaml"
    with open(config_path) as handle:
        return cast(dict[str, Any], yaml.safe_load(handle))


def discover(
    checkpoints_dir: Path = CHECKPOINTS_DIR,
    cultures: tuple[str, ...] = CULTURES,
    backbones: tuple[str, ...] = DEFAULT_MODELS,
    problem: str = DEFAULT_PROBLEM,
    config_dir: Path = CONFIG_DIR,
) -> list[Checkpoint]:
    found = []
    for culture in cultures:
        for backbone in backbones:
            ckpt_dir = checkpoints_dir / culture / backbone / problem
            if not (ckpt_dir / "TRAINING_DONE").is_file():
                continue
            if not (ckpt_dir / "adapter_model.safetensors").is_file():
                continue
            config = _load_config(config_dir, backbone, problem)
            base_model_id = str(config["model"]["id"])
            lora = cast(dict[str, Any], config.get("lora", {}))
            found.append(Checkpoint(culture, backbone, problem, ckpt_dir, base_model_id, lora))
    return found


def render_model_card(ckpt: Checkpoint) -> str:
    target_modules = ", ".join(f"`{m}`" for m in ckpt.lora.get("target_modules", []))
    return f"""---
base_model: {ckpt.base_model_id}
library_name: peft
tags:
- lora
- culture-mllm
- culture-steering
- culture:{ckpt.culture}
pipeline_tag: text-generation
---

# {ckpt.backbone} — {ckpt.culture} (culture-steering LoRA)

Part of the [Culture-Steering-MLLM-Collection]\
(https://huggingface.co/{DEFAULT_REPO_ID}), one adapter per
(culture, backbone) pair from
[culture-mllm](https://github.com/neemiasbsilva/culture-mllm).

This is a LoRA adapter for [{ckpt.base_model_id}](https://huggingface.co/{ckpt.base_model_id}),
fine-tuned on World Values Survey question/answer text under a
`{ckpt.culture}` system prompt, following the CultureLLM (Li et al.,
NeurIPS 2024) recipe extended into the multimodal domain.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("{ckpt.base_model_id}")
model = PeftModel.from_pretrained(base_model, "{DEFAULT_REPO_ID}", subfolder="{ckpt.path_in_repo}")
tokenizer = AutoTokenizer.from_pretrained("{DEFAULT_REPO_ID}", subfolder="{ckpt.path_in_repo}")
```

## LoRA configuration

- r: {ckpt.lora.get("r", "n/a")}
- alpha: {ckpt.lora.get("alpha", "n/a")}
- dropout: {ckpt.lora.get("dropout", "n/a")}
- target modules: {target_modules or "n/a"}

## License

Inherits the licensing terms of the base model,
[{ckpt.base_model_id}](https://huggingface.co/{ckpt.base_model_id}).

## Citation

See [culture-mllm](https://github.com/neemiasbsilva/culture-mllm) for the
accompanying paper (EMNLP 2026, in progress).
"""


def render_collection_card(checkpoints: list[Checkpoint], repo_id: str) -> str:
    backbones = sorted({c.backbone for c in checkpoints})
    cultures = sorted({c.culture for c in checkpoints})
    present = {(c.culture, c.backbone) for c in checkpoints}
    rows = "\n".join(
        f"| {culture} | "
        + " | ".join("✅" if (culture, b) in present else "" for b in backbones)
        + " |"
        for culture in cultures
    )
    header = "| culture | " + " | ".join(backbones) + " |"
    divider = "|---|" + "---|" * len(backbones)
    return f"""---
license: other
tags:
- culture-mllm
- culture-steering
- lora
- peft
---

# Culture-Steering MLLM Collection

LoRA adapters that fine-tune each backbone into ten cultures from World
Values Survey question/answer text, extending CultureLLM (Li et al.,
NeurIPS 2024) into the multimodal domain.
Source: [culture-mllm](https://github.com/neemiasbsilva/culture-mllm).

Every adapter lives under `{{culture}}/{{backbone}}/{{problem}}` in this repo,
e.g. `english/qwen3_5_2b/cultural`. Load with `PeftModel.from_pretrained(...,
subfolder="english/qwen3_5_2b/cultural")`.

## Coverage

{header}
{divider}
{rows}
"""


def status(
    repo_id: str = DEFAULT_REPO_ID,
    checkpoints_dir: Path = CHECKPOINTS_DIR,
    cultures: tuple[str, ...] = CULTURES,
    backbones: tuple[str, ...] = DEFAULT_MODELS,
    problem: str = DEFAULT_PROBLEM,
    config_dir: Path = CONFIG_DIR,
) -> None:
    checkpoints = discover(checkpoints_dir, cultures, backbones, problem, config_dir)
    api = HfApi()
    try:
        remote_files = set(api.list_repo_files(repo_id))
    except RepositoryNotFoundError:
        remote_files = set()

    pushed: list[Checkpoint] = []
    pending: list[Checkpoint] = []
    for ckpt in checkpoints:
        remote_marker = f"{ckpt.path_in_repo}/adapter_model.safetensors"
        (pushed if remote_marker in remote_files else pending).append(ckpt)

    print(f"Repo: {repo_id}")
    print(f"Local checkpoints found: {len(checkpoints)}")
    print(f"Already on the Hub     : {len(pushed)}")
    print(f"Pending                : {len(pending)}")
    for ckpt in pending:
        print(f"  pending: {ckpt.path_in_repo}")


def push(
    repo_id: str = DEFAULT_REPO_ID,
    checkpoints_dir: Path = CHECKPOINTS_DIR,
    cultures: tuple[str, ...] = CULTURES,
    backbones: tuple[str, ...] = DEFAULT_MODELS,
    problem: str = DEFAULT_PROBLEM,
    private: bool = False,
    dry_run: bool = False,
    config_dir: Path = CONFIG_DIR,
) -> None:
    checkpoints = discover(checkpoints_dir, cultures, backbones, problem, config_dir)
    if not checkpoints:
        print("No completed checkpoints matched the given cultures/backbones/problem.")
        return

    verb = "Would push" if dry_run else "Pushing"
    print(f"{verb} {len(checkpoints)} checkpoint(s) to {repo_id}:")
    for ckpt in checkpoints:
        print(f"  {ckpt.path_in_repo}  ({ckpt.base_model_id})")

    if dry_run:
        return

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)

    api.create_commit(
        repo_id=repo_id,
        operations=[
            CommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=render_collection_card(checkpoints, repo_id).encode(),
            )
        ],
        commit_message="Update collection README",
    )

    for ckpt in checkpoints:
        operations = [
            CommitOperationAdd(
                path_in_repo=f"{ckpt.path_in_repo}/{filename}",
                path_or_fileobj=str(ckpt.path / filename),
            )
            for filename in PUSH_FILENAMES
            if (ckpt.path / filename).is_file()
        ]
        operations.append(
            CommitOperationAdd(
                path_in_repo=f"{ckpt.path_in_repo}/README.md",
                path_or_fileobj=render_model_card(ckpt).encode(),
            )
        )
        api.create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message=f"Push {ckpt.path_in_repo}",
        )
        print(f"  done: {ckpt.path_in_repo}")

    print("Push complete.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    common.add_argument("--checkpoints-dir", type=Path, default=CHECKPOINTS_DIR)
    common.add_argument("--cultures", nargs="+", default=list(CULTURES))
    common.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), dest="backbones")
    common.add_argument("--problem", default=DEFAULT_PROBLEM)

    sub.add_parser("status", parents=[common])
    push_parser = sub.add_parser("push", parents=[common])
    push_parser.add_argument("--private", action="store_true")
    push_parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "status":
        status(
            repo_id=args.repo_id,
            checkpoints_dir=args.checkpoints_dir,
            cultures=tuple(args.cultures),
            backbones=tuple(args.backbones),
            problem=args.problem,
        )
    elif args.command == "push":
        push(
            repo_id=args.repo_id,
            checkpoints_dir=args.checkpoints_dir,
            cultures=tuple(args.cultures),
            backbones=tuple(args.backbones),
            problem=args.problem,
            private=args.private,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
