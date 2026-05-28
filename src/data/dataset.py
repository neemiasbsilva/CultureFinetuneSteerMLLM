"""
PyTorch Dataset and HuggingFace Dataset helpers for CultureVLM training.

Provides:
  - CultureVLMDataset: torch.utils.data.Dataset wrapping JSONL SFT examples
  - merge_wvs_and_visual: merge WVS text anchoring + visual examples into one split
  - load_hf_dataset: returns a HuggingFace datasets.Dataset for trl.SFTTrainer
"""

import json
from pathlib import Path

import torch
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset


class CultureVLMDataset(TorchDataset):
    """
    Loads a JSONL file of chat-completion examples (with optional image paths).
    Each record has {"condition", "culture", "image_id", "messages": [...]}.
    """

    def __init__(self, jsonl_path: str | Path):
        self.path = Path(jsonl_path)
        self.records: list[dict] = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]

    def __repr__(self) -> str:
        return f"CultureVLMDataset(path={self.path}, n={len(self)})"


def merge_wvs_and_visual(
    wvs_jsonl: str | Path,
    visual_jsonl: str | Path,
    output_jsonl: str | Path,
) -> int:
    """
    Merge WVS cultural anchoring examples + visual urban sentiment examples.
    Returns total number of merged examples.
    """
    records: list[dict] = []

    wvs_path = Path(wvs_jsonl)
    if wvs_path.exists():
        with open(wvs_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    vis_path = Path(visual_jsonl)
    if vis_path.exists():
        with open(vis_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(records)


def load_hf_dataset(jsonl_path: str | Path, text_field: str = "text") -> Dataset:
    """
    Load JSONL as a HuggingFace Dataset. For trl.SFTTrainer, the dataset
    should expose either a 'messages' field (chat-template mode) or a
    pre-formatted 'text' field.

    This function returns a Dataset with 'messages' column so that
    SFTTrainer can apply the model's chat template automatically.
    """
    records = []
    with open(Path(jsonl_path)) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                records.append({"messages": obj["messages"]})
    return Dataset.from_list(records)
