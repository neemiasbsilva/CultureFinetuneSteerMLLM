#!/usr/bin/env python3
"""Summarize groups, workers, and assignments in data/raw/final.json.

By default this script treats explicit labels like ``group9_form8`` as groups,
using the ``group9`` prefix. Use ``--mode worker`` to instead summarize unique
``worker_id`` values.

Usage:
    python3 scripts/analyze_final_groups.py
    python3 scripts/analyze_final_groups.py --mode worker
    python3 scripts/analyze_final_groups.py --csv outputs/analysis/final_group_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "final.json"
GROUP_RE = re.compile(r"^(group\d+)(?:_form\d+)?$")


@dataclass
class GroupSummary:
    """Aggregated task statistics for one group key."""

    tasks: int = 0
    image_responses: int = 0
    assignment_ids: set[str] = field(default_factory=set)
    worker_ids: set[str] = field(default_factory=set)
    image_ids: set[str] = field(default_factory=set)

    def add_task(self, task: dict[str, Any]) -> None:
        """Add one final.json task to the summary."""
        self.tasks += 1

        assignment_id = str(task.get("assignment_id") or "")
        worker_id = str(task.get("worker_id") or "")
        if assignment_id:
            self.assignment_ids.add(assignment_id)
        if worker_id:
            self.worker_ids.add(worker_id)

        image_resps = task.get("image_resps") or []
        if isinstance(image_resps, list):
            self.image_responses += len(image_resps)
            for image_resp in image_resps:
                if isinstance(image_resp, dict) and image_resp.get("id"):
                    self.image_ids.add(str(image_resp["id"]))


def load_tasks(path: Path) -> list[dict[str, Any]]:
    """Load tasks from the final.json wrapper object."""
    with open(path) as f:
        payload = json.load(f)

    tasks = payload.get("tarefas")
    if not isinstance(tasks, list):
        raise ValueError(f"Expected {path} to contain a top-level 'tarefas' list.")
    return tasks


def group_key(task: dict[str, Any], mode: str) -> str | None:
    """Return the grouping key for one task."""
    assignment_id = str(task.get("assignment_id") or "")
    worker_id = str(task.get("worker_id") or "")

    if mode == "worker":
        return worker_id or None

    if mode == "assignment-prefix":
        if "_form" in assignment_id:
            return assignment_id.split("_form", maxsplit=1)[0]
        return assignment_id or None

    assignment_match = GROUP_RE.match(assignment_id)
    if assignment_match:
        return assignment_match.group(1)

    worker_match = GROUP_RE.match(worker_id)
    if worker_match:
        return worker_match.group(1)

    return None


def summarize(tasks: list[dict[str, Any]], mode: str) -> tuple[dict[str, GroupSummary], int]:
    """Build per-group summaries and count tasks without a group key."""
    groups: dict[str, GroupSummary] = defaultdict(GroupSummary)
    ungrouped_tasks = 0

    for task in tasks:
        key = group_key(task, mode)
        if key is None:
            ungrouped_tasks += 1
            continue
        groups[key].add_task(task)

    return dict(groups), ungrouped_tasks


def natural_group_sort(key: str) -> tuple[int, int | str]:
    """Sort group labels like group9 before group10, then other labels."""
    match = re.fullmatch(r"group(\d+)", key)
    if match:
        return 0, int(match.group(1))
    return 1, key


def clipped(values: set[str], limit: int) -> str:
    """Return a comma-separated preview of set values."""
    ordered = sorted(values, key=natural_group_sort)
    if len(ordered) <= limit:
        return ", ".join(ordered)
    shown = ", ".join(ordered[:limit])
    return f"{shown}, ... (+{len(ordered) - limit} more)"


def print_summary(
    path: Path,
    tasks: list[dict[str, Any]],
    groups: dict[str, GroupSummary],
    ungrouped_tasks: int,
    mode: str,
    max_list: int,
) -> None:
    """Print a compact terminal report."""
    grouped_tasks = sum(summary.tasks for summary in groups.values())
    total_image_responses = sum(summary.image_responses for summary in groups.values())
    total_unique_images = len(set().union(*(summary.image_ids for summary in groups.values()))) if groups else 0
    worker_ids = {
        str(task.get("worker_id") or "")
        for task in tasks
        if task.get("worker_id")
    }
    group_worker_ids = {worker_id for worker_id in worker_ids if GROUP_RE.match(worker_id)}
    non_group_worker_ids = worker_ids - group_worker_ids

    print(f"Input: {path}")
    print(f"Mode: {mode}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Grouped tasks: {grouped_tasks}")
    print(f"Ungrouped tasks: {ungrouped_tasks}")
    print(f"Unique groups: {len(groups)}")
    print(f"Unique worker_id values: {len(worker_ids)}")
    print(f"Unique worker_id values in group id format: {len(group_worker_ids)}")
    print(f"Unique worker_id values absent from group id format: {len(non_group_worker_ids)}")
    print(f"Image responses in grouped tasks: {total_image_responses}")
    print(f"Unique images in grouped tasks: {total_unique_images}")
    print()

    header = (
        "group",
        "tasks",
        "workers",
        "assignments",
        "image_resps",
        "unique_images",
        "associated_workers",
    )
    rows = []
    for key in sorted(groups, key=natural_group_sort):
        summary = groups[key]
        rows.append(
            (
                key,
                str(summary.tasks),
                str(len(summary.worker_ids)),
                str(len(summary.assignment_ids)),
                str(summary.image_responses),
                str(len(summary.image_ids)),
                clipped(summary.worker_ids, max_list),
            )
        )

    widths = [len(column) for column in header]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row, strict=True)]

    print("  ".join(column.ljust(width) for column, width in zip(header, widths, strict=True)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths, strict=True)))


def write_csv(path: Path, groups: dict[str, GroupSummary]) -> None:
    """Write complete per-group details to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "tasks",
                "unique_workers",
                "unique_assignments",
                "image_responses",
                "unique_images",
                "worker_ids",
                "assignment_ids",
                "image_ids",
            ],
        )
        writer.writeheader()
        for key in sorted(groups, key=natural_group_sort):
            summary = groups[key]
            writer.writerow(
                {
                    "group": key,
                    "tasks": summary.tasks,
                    "unique_workers": len(summary.worker_ids),
                    "unique_assignments": len(summary.assignment_ids),
                    "image_responses": summary.image_responses,
                    "unique_images": len(summary.image_ids),
                    "worker_ids": "|".join(sorted(summary.worker_ids, key=natural_group_sort)),
                    "assignment_ids": "|".join(sorted(summary.assignment_ids)),
                    "image_ids": "|".join(sorted(summary.image_ids)),
                }
            )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze unique groups and associated workers in final.json."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_INPUT,
        type=Path,
        help="Path to final.json. Defaults to data/raw/final.json.",
    )
    parser.add_argument(
        "--mode",
        choices=("group", "worker", "assignment-prefix"),
        default="group",
        help=(
            "Grouping strategy. 'group' extracts groupNN labels from assignment_id or worker_id; "
            "'worker' summarizes unique worker_id values; 'assignment-prefix' groups by the "
            "assignment_id prefix before _form, otherwise the whole assignment_id."
        ),
    )
    parser.add_argument(
        "--max-list",
        type=int,
        default=8,
        help="Maximum associated worker IDs to show in the terminal table.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path with complete workers, assignments, and image IDs.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the analysis."""
    args = parse_args()
    input_path = args.input.expanduser().resolve()

    tasks = load_tasks(input_path)
    groups, ungrouped_tasks = summarize(tasks, args.mode)
    print_summary(input_path, tasks, groups, ungrouped_tasks, args.mode, args.max_list)

    if args.csv:
        csv_path = args.csv.expanduser().resolve()
        write_csv(csv_path, groups)
        print()
        print(f"Wrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
