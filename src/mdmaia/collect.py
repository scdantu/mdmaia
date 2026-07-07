"""Collect local ligand/cofactor coordinates from trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .io import write_table
from .pbc import Box, distances, nearest_image_vector

COLLECT_COLUMNS = [
    "frame",
    "time_ps",
    "target_label",
    "target_selection",
    "mobile_selection",
    "mobile_index",
    "mobile_resid",
    "mobile_resname",
    "mobile_name",
    "x",
    "y",
    "z",
    "distance",
]


@dataclass
class CollectConfig:
    topology: str
    trajectory: str
    target: str
    mobile: str
    output: str | None = None
    cutoff: float | None = None
    start: int | None = None
    stop: int | None = None
    step: int | None = None
    target_label: str = "target"


def collect_local_positions(config: CollectConfig) -> pd.DataFrame:
    """Collect nearest-image positions of a mobile selection around a target.

    The target is represented by its centre of geometry. Each atom in the mobile
    selection is recorded independently so ions, waters and ligand atoms can all
    be handled with the same machinery.
    """

    try:
        import MDAnalysis as mda
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError("MDAnalysis is required for trajectory collection.") from exc

    universe = mda.Universe(config.topology, config.trajectory)
    target = universe.select_atoms(config.target)
    mobile = universe.select_atoms(config.mobile)
    if len(target) == 0:
        raise ValueError(f"Target selection matched no atoms: {config.target!r}")
    if len(mobile) == 0:
        raise ValueError(f"Mobile selection matched no atoms: {config.mobile!r}")

    rows: list[dict[str, object]] = []
    frames = universe.trajectory[config.start : config.stop : config.step]
    for ts in frames:
        target_xyz = target.center_of_geometry()
        box = Box.from_dimensions(ts.dimensions)
        vecs = nearest_image_vector(target_xyz, mobile.positions, box)
        dists = distances(vecs)

        for atom, vec, dist in zip(mobile.atoms, vecs, dists):
            if config.cutoff is not None and dist > config.cutoff:
                continue
            rows.append(
                {
                    "frame": int(ts.frame),
                    "time_ps": float(ts.time),
                    "target_label": config.target_label,
                    "target_selection": config.target,
                    "mobile_selection": config.mobile,
                    "mobile_index": int(atom.index),
                    "mobile_resid": int(atom.resid),
                    "mobile_resname": str(atom.resname),
                    "mobile_name": str(atom.name),
                    "x": float(vec[0]),
                    "y": float(vec[1]),
                    "z": float(vec[2]),
                    "distance": float(dist),
                }
            )

    df = pd.DataFrame(rows, columns=COLLECT_COLUMNS)
    if config.output:
        write_table(df, config.output)
    return df


def collect_from_args(args) -> pd.DataFrame:
    config = CollectConfig(
        topology=args.topology,
        trajectory=args.trajectory,
        target=args.target,
        mobile=args.mobile,
        output=args.output,
        cutoff=args.cutoff,
        start=args.start,
        stop=args.stop,
        step=args.step,
        target_label=args.target_label,
    )
    return collect_local_positions(config)
