"""Hotspot/site detection utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .io import read_table, write_table


def cluster_sites(
    df: pd.DataFrame,
    eps: float = 1.0,
    min_samples: int = 20,
) -> pd.DataFrame:
    """Cluster local-coordinate points into putative ligand/cofactor sites."""

    if df.empty:
        return pd.DataFrame(columns=["site_id", "x", "y", "z", "n_points", "fraction"])
    coords = df[["x", "y", "z"]].to_numpy(dtype=float)
    labels = _radius_cluster(coords, eps=eps, min_samples=min_samples)
    return summarize_clusters(df, labels)


def assign_clusters(
    df: pd.DataFrame,
    eps: float = 1.0,
    min_samples: int = 20,
) -> pd.DataFrame:
    """Return input contact rows with a ``cluster_id`` column."""

    if df.empty:
        out = df.copy()
        out["cluster_id"] = pd.Series(dtype=int)
        return out
    coords = df[["x", "y", "z"]].to_numpy(dtype=float)
    labels = _radius_cluster(coords, eps=eps, min_samples=min_samples)
    out = df.copy()
    out["cluster_id"] = labels
    return out


def summarize_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Summarise cluster centres and point counts."""

    if df.empty:
        return pd.DataFrame(columns=["site_id", "x", "y", "z", "n_points", "fraction"])
    coords = df[["x", "y", "z"]].to_numpy(dtype=float)
    rows = []
    total = len(labels)
    for label in sorted(set(labels)):
        if label == -1:
            continue
        idx = labels == label
        centre = coords[idx].mean(axis=0)
        rows.append(
            {
                "site_id": int(label),
                "x": float(centre[0]),
                "y": float(centre[1]),
                "z": float(centre[2]),
                "n_points": int(idx.sum()),
                "fraction": float(idx.sum() / total),
            }
        )
    return pd.DataFrame(rows).sort_values("fraction", ascending=False).reset_index(drop=True)


def _radius_cluster(coords: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Small DBSCAN-like clustering using only SciPy.

    Points with fewer than ``min_samples`` neighbours within ``eps`` are marked
    as noise unless they are density-reachable from a core point.
    """

    tree = cKDTree(coords)
    neighbours = tree.query_ball_point(coords, r=eps)
    labels = np.full(coords.shape[0], -1, dtype=int)
    visited = np.zeros(coords.shape[0], dtype=bool)
    cluster_id = 0

    for i in range(coords.shape[0]):
        if visited[i]:
            continue
        visited[i] = True
        if len(neighbours[i]) < min_samples:
            continue

        labels[i] = cluster_id
        seeds = list(neighbours[i])
        seed_set = set(seeds)
        cursor = 0
        while cursor < len(seeds):
            j = seeds[cursor]
            if not visited[j]:
                visited[j] = True
                if len(neighbours[j]) >= min_samples:
                    for candidate in neighbours[j]:
                        if candidate not in seed_set:
                            seeds.append(candidate)
                            seed_set.add(candidate)
            if labels[j] == -1:
                labels[j] = cluster_id
            cursor += 1
        cluster_id += 1

    return labels


def cluster_sites_file(
    input_path: str,
    output_path: str,
    eps: float = 1.0,
    min_samples: int = 20,
    assign_output: str | None = None,
) -> pd.DataFrame:
    df = read_table(input_path)
    assigned = assign_clusters(df, eps=eps, min_samples=min_samples)
    result = summarize_clusters(df, assigned["cluster_id"].to_numpy(dtype=int))
    write_table(result, output_path)
    if assign_output:
        write_table(assigned, assign_output)
    return result
