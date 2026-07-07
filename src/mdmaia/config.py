"""Configuration-driven MD-MAIA workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .collect import CollectConfig, collect_local_positions
from .io import write_table


@dataclass
class TargetSite:
    label: str
    selection: str
    cutoff: float | None = None


@dataclass
class CollectionWorkflow:
    topology: str
    trajectory: str
    mobile: str
    targets: list[TargetSite]
    output: str
    start: int | None = None
    stop: int | None = None
    step: int | None = None


def read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def parse_collection_config(path: str | Path) -> CollectionWorkflow:
    path = Path(path)
    base_dir = path.parent
    data = read_config(path)
    targets = [
        TargetSite(
            label=str(item["label"]),
            selection=str(item["selection"]),
            cutoff=item.get("cutoff", data.get("cutoff")),
        )
        for item in data["targets"]
    ]
    return CollectionWorkflow(
        topology=str(_resolve_path(base_dir, data["topology"])),
        trajectory=str(_resolve_path(base_dir, data["trajectory"])),
        mobile=str(data["mobile"]),
        targets=targets,
        output=str(_resolve_path(base_dir, data["output"])),
        start=data.get("start"),
        stop=data.get("stop"),
        step=data.get("step"),
    )


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def collect_from_config(path: str | Path) -> pd.DataFrame:
    workflow = parse_collection_config(path)
    frames = []
    for target in workflow.targets:
        config = CollectConfig(
            topology=workflow.topology,
            trajectory=workflow.trajectory,
            target=target.selection,
            mobile=workflow.mobile,
            output=None,
            cutoff=target.cutoff,
            start=workflow.start,
            stop=workflow.stop,
            step=workflow.step,
            target_label=target.label,
        )
        frames.append(collect_local_positions(config))
    non_empty = [frame for frame in frames if not frame.empty]
    result = (
        pd.concat(non_empty, ignore_index=True)
        if non_empty
        else pd.DataFrame(columns=frames[0].columns if frames else None)
    )
    write_table(result, workflow.output)
    return result
