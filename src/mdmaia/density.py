"""Density-grid generation from MD-MAIA local-coordinate tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io import read_table, write_dx


@dataclass
class DensityResult:
    grid: np.ndarray
    origin: np.ndarray
    spacing: float
    edges: tuple[np.ndarray, np.ndarray, np.ndarray]


def density_from_table(
    df: pd.DataFrame,
    spacing: float = 0.5,
    radius: float | None = None,
    normalize: bool = True,
) -> DensityResult:
    coords = df[["x", "y", "z"]].to_numpy(dtype=float)
    if coords.size == 0:
        raise ValueError("No coordinates available for density calculation.")

    if radius is None:
        max_abs = np.ceil(np.max(np.abs(coords)) / spacing) * spacing
        radius = float(max_abs)

    edges = tuple(np.arange(-radius, radius + spacing, spacing) for _ in range(3))
    grid, edges_out = np.histogramdd(coords, bins=edges)
    if normalize and grid.sum() > 0:
        grid = grid / grid.sum()
    origin = np.array([edge[0] for edge in edges_out], dtype=float)
    return DensityResult(grid=grid, origin=origin, spacing=spacing, edges=edges_out)


def density_file(
    input_path: str,
    output_path: str,
    spacing: float = 0.5,
    radius: float | None = None,
    normalize: bool = True,
) -> DensityResult:
    df = read_table(input_path)
    result = density_from_table(df, spacing=spacing, radius=radius, normalize=normalize)
    write_dx(result.grid, result.origin, result.spacing, output_path)
    return result
