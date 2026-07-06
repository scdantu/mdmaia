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
        topology=str(data["topology"]),
        trajectory=str(data["trajectory"]),
        mobile=str(data["mobile"]),
        targets=targets,
        output=str(data["output"]),
        start=data.get("start"),
        stop=data.get("stop"),
        step=data.get("step"),
    )


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
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    write_table(result, workflow.output)
    return result
