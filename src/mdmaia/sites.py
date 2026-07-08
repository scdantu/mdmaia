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


def consensus_sites(
    df: pd.DataFrame,
    eps: float = 3.0,
    min_samples: int = 1,
    by_condition: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster local hotspot centroids into consensus sites.

    The input should contain one row per local hotspot/cluster with ``x,y,z`` in
    a shared coordinate frame. If ``condition`` and ``replica`` columns are
    present, they are summarised in the consensus output.
    """

    if df.empty:
        empty_summary = pd.DataFrame(
            columns=[
                "consensus_site_id",
                "x",
                "y",
                "z",
                "n_local_clusters",
                "n_conditions",
                "n_replicas",
                "n_points_total",
                "dominant_target_label",
                "mean_occupancy",
                "max_residence_ps",
            ]
        )
        empty_mapping = df.copy()
        empty_mapping["consensus_site_id"] = pd.Series(dtype=int)
        return empty_summary, empty_mapping

    group_cols = ["condition"] if by_condition and "condition" in df.columns else []
    grouped = df.groupby(group_cols, observed=True) if group_cols else [((), df)]
    summary_parts = []
    mapping_parts = []
    site_offset = 0
    for key, group in grouped:
        coords = group[["x", "y", "z"]].to_numpy(dtype=float)
        labels = _radius_cluster(coords, eps=eps, min_samples=min_samples)
        assigned = group.copy()
        assigned["consensus_site_id"] = labels
        valid_labels = [label for label in sorted(set(labels)) if label >= 0]
        id_map = {label: site_offset + i for i, label in enumerate(valid_labels)}
        assigned["consensus_site_id"] = assigned["consensus_site_id"].map(
            lambda label: id_map.get(label, -1)
        )
        mapping_parts.append(assigned)
        for label in valid_labels:
            consensus_id = id_map[label]
            subset = assigned[assigned["consensus_site_id"] == consensus_id]
            weights = (
                subset["n_points"].to_numpy(dtype=float)
                if "n_points" in subset.columns
                else np.ones(len(subset), dtype=float)
            )
            xyz = subset[["x", "y", "z"]].to_numpy(dtype=float)
            centre = np.average(xyz, axis=0, weights=weights)
            row = {
                "consensus_site_id": int(consensus_id),
                "x": float(centre[0]),
                "y": float(centre[1]),
                "z": float(centre[2]),
                "n_local_clusters": int(len(subset)),
                "n_conditions": int(subset["condition"].nunique()) if "condition" in subset.columns else None,
                "n_replicas": int(subset["replica"].nunique()) if "replica" in subset.columns else None,
                "n_points_total": int(subset["n_points"].sum()) if "n_points" in subset.columns else int(len(subset)),
                "dominant_target_label": _dominant_value(subset, "dominant_target_label"),
                "mean_occupancy": float(subset["occupancy"].mean()) if "occupancy" in subset.columns else None,
                "max_residence_ps": float(subset["max_residence_ps"].max()) if "max_residence_ps" in subset.columns else None,
            }
            if group_cols:
                key_values = key if isinstance(key, tuple) else (key,)
                row.update(dict(zip(group_cols, key_values)))
            summary_parts.append(row)
        site_offset += len(valid_labels)

    summary = pd.DataFrame(summary_parts)
    mapping = pd.concat(mapping_parts, ignore_index=True) if mapping_parts else pd.DataFrame()
    if not summary.empty:
        summary = summary.sort_values(["n_replicas", "n_local_clusters", "n_points_total"], ascending=False)
    return summary, mapping


def _dominant_value(df: pd.DataFrame, column: str) -> str | None:
    if column not in df.columns or df[column].dropna().empty:
        return None
    return str(df[column].value_counts().idxmax())


def consensus_sites_file(
    input_path: str,
    output_path: str,
    eps: float = 3.0,
    min_samples: int = 1,
    mapping_output: str | None = None,
    by_condition: bool = False,
) -> pd.DataFrame:
    summary, mapping = consensus_sites(
        read_table(input_path),
        eps=eps,
        min_samples=min_samples,
        by_condition=by_condition,
    )
    write_table(summary, output_path)
    if mapping_output:
        write_table(mapping, mapping_output)
    return summary
