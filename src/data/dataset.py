"""PyTorch Dataset and HuggingFace Dataset helpers for CultureVLM training."""

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, Sequence
from datasets import Image as HFImage
from torch.utils.data import Dataset as TorchDataset


class CultureVLMDataset(TorchDataset[dict[str, Any]]):
    def __init__(self, jsonl_path: str | Path):
        self.path = Path(jsonl_path)
        self.records: list[dict[str, Any]] = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]

    def __repr__(self) -> str:
        return f"CultureVLMDataset(path={self.path}, n={len(self)})"


def merge_wvs_and_visual(
    wvs_jsonl: str | Path,
    visual_jsonl: str | Path,
    output_jsonl: str | Path,
) -> int:
    records: list[dict[str, Any]] = []

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
    records = []
    with open(Path(jsonl_path)) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                rec = {"messages": obj["messages"]}
                if obj.get("images"):
                    rec["images"] = obj["images"]
                records.append(rec)
    dataset = Dataset.from_list(records)
    if "images" in dataset.column_names:
        dataset = dataset.cast_column("images", Sequence(HFImage()))
    return dataset
