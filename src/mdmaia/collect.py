"""Collect local ligand/cofactor coordinates from trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
    "condition",
    "replica",
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
    coordinates: str = "local"
    align_selection: str | None = None
    condition: str | None = None
    replica: str | None = None


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
    if config.coordinates not in {"local", "aligned"}:
        raise ValueError("coordinates must be either 'local' or 'aligned'")
    if config.coordinates == "aligned" and not config.align_selection:
        raise ValueError("aligned coordinates require align_selection")

    align_group = None
    reference_align = None
    first_frame = config.start if config.start is not None else 0
    if config.align_selection:
        align_group = universe.select_atoms(config.align_selection)
        if len(align_group) == 0:
            raise ValueError(f"Align selection matched no atoms: {config.align_selection!r}")
        universe.trajectory[first_frame]
        reference_align = align_group.positions.copy()

    rows: list[dict[str, object]] = []
    frames = universe.trajectory[config.start : config.stop : config.step]
    for ts in frames:
        target_xyz = target.center_of_geometry()
        box = Box.from_dimensions(ts.dimensions)
        vecs = nearest_image_vector(target_xyz, mobile.positions, box)
        dists = distances(vecs)
        coords = vecs
        if config.coordinates == "aligned":
            nearest_positions = target_xyz + vecs
            coords = _align_coordinates(
                nearest_positions,
                align_group.positions,
                reference_align,
            )

        for atom, coord, dist in zip(mobile.atoms, coords, dists):
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
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "z": float(coord[2]),
                    "distance": float(dist),
                    "condition": config.condition,
                    "replica": config.replica,
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
        coordinates=args.coordinates,
        align_selection=args.align_selection,
        condition=args.condition,
        replica=args.replica,
    )
    return collect_local_positions(config)


def _align_coordinates(
    coordinates: np.ndarray,
    mobile_reference_atoms: np.ndarray,
    fixed_reference_atoms: np.ndarray,
) -> np.ndarray:
    """Rigidly align coordinates from current frame onto the reference frame."""

    mobile = np.asarray(mobile_reference_atoms, dtype=float)
    fixed = np.asarray(fixed_reference_atoms, dtype=float)
    coords = np.asarray(coordinates, dtype=float)

    mobile_center = mobile.mean(axis=0)
    fixed_center = fixed.mean(axis=0)
    mobile0 = mobile - mobile_center
    fixed0 = fixed - fixed_center

    covariance = mobile0.T @ fixed0
    left, _, right_t = np.linalg.svd(covariance)
    handedness = np.sign(np.linalg.det(left @ right_t))
    correction = np.diag([1.0, 1.0, handedness])
    rotation = left @ correction @ right_t
    return (coords - mobile_center) @ rotation + fixed_center
