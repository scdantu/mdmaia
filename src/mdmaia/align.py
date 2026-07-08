"""Coordinate-frame alignment utilities for comparing site centroids."""

from __future__ import annotations

from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd

from .io import read_table, write_table


def fit_transform(mobile_xyz: np.ndarray, reference_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return rotation and translation that fit mobile coordinates onto reference.

    The returned values transform row-vector coordinates as ``xyz @ rotation + translation``.
    """

    mobile = np.asarray(mobile_xyz, dtype=float)
    reference = np.asarray(reference_xyz, dtype=float)
    if mobile.shape != reference.shape:
        raise ValueError(
            f"Alignment selections must have matching shapes; got {mobile.shape} and {reference.shape}."
        )
    if mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("Alignment coordinates must be an N x 3 array.")
    if mobile.shape[0] < 3:
        raise ValueError("At least three atoms are required for a stable fit.")

    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)
    mobile0 = mobile - mobile_center
    reference0 = reference - reference_center

    covariance = mobile0.T @ reference0
    v, _, wt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(v @ wt))
    rotation = v @ correction @ wt
    translation = reference_center - mobile_center @ rotation
    return rotation, translation


def transform_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Apply a row-vector rigid-body transform to points."""

    return np.asarray(points, dtype=float) @ rotation + np.asarray(translation, dtype=float)


def centroid_transform(
    mobile_structure: str | Path,
    reference_structure: str | Path,
    selection: str,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Calculate transform from a centroid PDB onto a reference PDB."""

    mobile_universe = mda.Universe(str(mobile_structure))
    reference_universe = mda.Universe(str(reference_structure))
    mobile_atoms = mobile_universe.select_atoms(selection)
    reference_atoms = reference_universe.select_atoms(selection)
    if len(mobile_atoms) == 0:
        raise ValueError(f"No atoms matched selection in {mobile_structure!s}: {selection!r}")
    if len(reference_atoms) == 0:
        raise ValueError(f"No atoms matched selection in {reference_structure!s}: {selection!r}")

    rotation, translation = fit_transform(mobile_atoms.positions, reference_atoms.positions)
    fitted = transform_points(mobile_atoms.positions, rotation, translation)
    rmsd = float(np.sqrt(np.mean(np.sum((fitted - reference_atoms.positions) ** 2, axis=1))))
    return rotation, translation, rmsd, len(mobile_atoms)


def align_site_centroids(
    df: pd.DataFrame,
    *,
    structure_column: str,
    reference_structure: str | Path,
    selection: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transform site centroid rows to a common reference-centroid frame."""

    required = {"x", "y", "z", structure_column}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Input table is missing required columns: {', '.join(missing)}")

    transformed_parts = []
    transform_rows = []
    for structure, group in df.groupby(structure_column, observed=True):
        rotation, translation, rmsd, n_atoms = centroid_transform(
            structure,
            reference_structure,
            selection,
        )
        xyz = group[["x", "y", "z"]].to_numpy(dtype=float)
        fitted_xyz = transform_points(xyz, rotation, translation)
        out = group.copy()
        out[["x_original", "y_original", "z_original"]] = out[["x", "y", "z"]]
        out[["x", "y", "z"]] = fitted_xyz
        out["reference_structure"] = str(reference_structure)
        out["alignment_rmsd"] = rmsd
        out["alignment_n_atoms"] = n_atoms
        transformed_parts.append(out)
        transform_rows.append(
            {
                structure_column: structure,
                "reference_structure": str(reference_structure),
                "selection": selection,
                "alignment_rmsd": rmsd,
                "alignment_n_atoms": n_atoms,
                "rotation_00": rotation[0, 0],
                "rotation_01": rotation[0, 1],
                "rotation_02": rotation[0, 2],
                "rotation_10": rotation[1, 0],
                "rotation_11": rotation[1, 1],
                "rotation_12": rotation[1, 2],
                "rotation_20": rotation[2, 0],
                "rotation_21": rotation[2, 1],
                "rotation_22": rotation[2, 2],
                "translation_x": translation[0],
                "translation_y": translation[1],
                "translation_z": translation[2],
            }
        )

    transformed = pd.concat(transformed_parts, ignore_index=True) if transformed_parts else df.copy()
    transforms = pd.DataFrame(transform_rows)
    return transformed, transforms


def align_site_centroids_file(
    input_path: str,
    output_path: str,
    *,
    structure_column: str,
    reference_structure: str,
    selection: str,
    transforms_output: str | None = None,
) -> pd.DataFrame:
    """File wrapper for :func:`align_site_centroids`."""

    transformed, transforms = align_site_centroids(
        read_table(input_path),
        structure_column=structure_column,
        reference_structure=reference_structure,
        selection=selection,
    )
    write_table(transformed, output_path)
    if transforms_output:
        write_table(transforms, transforms_output)
    return transformed
