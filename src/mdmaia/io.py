"""Input/output helpers for MD-MAIA tables and grids."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_dx(
    grid: np.ndarray,
    origin: np.ndarray,
    spacing: float,
    path: str | Path,
    object_name: str = "density",
) -> None:
    """Write a simple OpenDX scalar grid."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx, ny, nz = grid.shape
    ox, oy, oz = origin
    with path.open("w") as handle:
        handle.write(f"# {object_name}\n")
        handle.write(f"object 1 class gridpositions counts {nx} {ny} {nz}\n")
        handle.write(f"origin {ox:.6f} {oy:.6f} {oz:.6f}\n")
        handle.write(f"delta {spacing:.6f} 0 0\n")
        handle.write(f"delta 0 {spacing:.6f} 0\n")
        handle.write(f"delta 0 0 {spacing:.6f}\n")
        handle.write(f"object 2 class gridconnections counts {nx} {ny} {nz}\n")
        handle.write(f"object 3 class array type double rank 0 items {grid.size} data follows\n")
        flat = grid.ravel(order="C")
        for i in range(0, flat.size, 3):
            handle.write(" ".join(f"{x:.8e}" for x in flat[i : i + 3]) + "\n")
        handle.write('attribute "dep" string "positions"\n')
        handle.write('object "density" class field\n')
        handle.write('component "positions" value 1\n')
        handle.write('component "connections" value 2\n')
        handle.write('component "data" value 3\n')
