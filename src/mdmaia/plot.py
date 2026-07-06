"""Plotting helpers for MD-MAIA outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io import read_table


def plot_occupancy(input_path: str, output_path: str) -> None:
    df = read_table(input_path)
    if "occupancy" not in df.columns:
        raise ValueError("Input table must contain an 'occupancy' column.")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(5, 0.35 * len(df)), 4))
    ax.bar(df["target_label"].astype(str), df["occupancy"], color="#4477AA")
    ax.set_ylabel("Occupancy")
    ax.set_xlabel("Target site")
    ax.set_ylim(0, max(1.0, float(df["occupancy"].max()) * 1.1))
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_cooccupancy(input_path: str, output_path: str) -> None:
    df = read_table(input_path)
    if "site" in df.columns:
        matrix = df.set_index("site")
    else:
        matrix = df.set_index(df.columns[0])
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=60, ha="right")
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    ax.set_title("Co-occupancy probability")
    fig.colorbar(image, ax=ax, label="Probability")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_feature_matrix(input_path: str, output_path: str) -> None:
    """Write numeric feature columns as a NumPy array."""

    df = read_table(input_path)
    numeric = df.select_dtypes(include="number")
    import numpy as np

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, numeric.to_numpy())
