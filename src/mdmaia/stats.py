"""Occupancy, co-occupancy and residence statistics."""

from __future__ import annotations

import pandas as pd

from .io import read_table, write_table


def occupancy_by_target(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise occupancy of each target by frame.

    Assumes the input table is already filtered to the relevant cutoff, either
    during ``collect`` or before calling this function.
    """

    if df.empty:
        return pd.DataFrame(
            columns=[
                "target_label",
                "n_frames_total",
                "n_frames_occupied",
                "occupancy",
                "mean_mobile_count",
                "max_mobile_count",
            ]
        )

    frame_counts = (
        df.groupby(["target_label", "frame"], observed=True)["mobile_index"]
        .nunique()
        .reset_index(name="mobile_count")
    )
    total_frames = df["frame"].nunique()
    out = (
        frame_counts.groupby("target_label", observed=True)
        .agg(
            n_frames_occupied=("frame", "nunique"),
            mean_mobile_count=("mobile_count", "mean"),
            max_mobile_count=("mobile_count", "max"),
        )
        .reset_index()
    )
    out["n_frames_total"] = total_frames
    out["occupancy"] = out["n_frames_occupied"] / total_frames
    return out[
        [
            "target_label",
            "n_frames_total",
            "n_frames_occupied",
            "occupancy",
            "mean_mobile_count",
            "max_mobile_count",
        ]
    ]


def multi_occupancy(df: pd.DataFrame) -> pd.DataFrame:
    """Probability of observing at least N mobile atoms per frame."""

    if df.empty:
        return pd.DataFrame(columns=["threshold", "n_frames", "probability"])
    counts = df.groupby("frame")["mobile_index"].nunique()
    n_frames = counts.shape[0]
    rows = []
    for threshold in range(1, int(counts.max()) + 1):
        hits = int((counts >= threshold).sum())
        rows.append({"threshold": threshold, "n_frames": hits, "probability": hits / n_frames})
    return pd.DataFrame(rows)


def cooccupancy_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Frame-wise site co-occupancy probability matrix.

    Values are ``P(site_i and site_j occupied)`` over frames represented in the
    input table. Use a pre-filtered collection table for a specific cutoff.
    """

    if df.empty:
        return pd.DataFrame()
    presence = (
        df.assign(occupied=1)
        .pivot_table(
            index="frame",
            columns="target_label",
            values="occupied",
            aggfunc="max",
            fill_value=0,
            observed=True,
        )
        .astype(int)
    )
    return presence.T.dot(presence) / len(presence)


def conditional_occupancy(df: pd.DataFrame) -> pd.DataFrame:
    """Conditional probability ``P(site_b occupied | site_a occupied)``."""

    if df.empty:
        return pd.DataFrame(columns=["site_a", "site_b", "p_b_given_a"])
    presence = (
        df.assign(occupied=1)
        .pivot_table(
            index="frame",
            columns="target_label",
            values="occupied",
            aggfunc="max",
            fill_value=0,
            observed=True,
        )
        .astype(int)
    )
    rows = []
    for site_a in presence.columns:
        denom = presence[site_a].sum()
        for site_b in presence.columns:
            prob = 0.0 if denom == 0 else float(((presence[site_a] == 1) & (presence[site_b] == 1)).sum() / denom)
            rows.append({"site_a": site_a, "site_b": site_b, "p_b_given_a": prob})
    return pd.DataFrame(rows)


def residence_times(df: pd.DataFrame, frame_step_ps: float | None = None) -> pd.DataFrame:
    """Consecutive occupied-frame runs per target and mobile object."""

    if df.empty:
        return pd.DataFrame(
            columns=[
                "target_label",
                "mobile_index",
                "start_frame",
                "end_frame",
                "n_frames",
                "duration_ps",
            ]
        )

    rows = []
    for (target, mobile), group in df.groupby(["target_label", "mobile_index"], observed=True):
        frames = sorted(set(int(x) for x in group["frame"]))
        if not frames:
            continue
        start = prev = frames[0]
        for frame in frames[1:]:
            if frame == prev + 1:
                prev = frame
                continue
            n = prev - start + 1
            rows.append(
                {
                    "target_label": target,
                    "mobile_index": mobile,
                    "start_frame": start,
                    "end_frame": prev,
                    "n_frames": n,
                    "duration_ps": None if frame_step_ps is None else n * frame_step_ps,
                }
            )
            start = prev = frame
        n = prev - start + 1
        rows.append(
            {
                "target_label": target,
                "mobile_index": mobile,
                "start_frame": start,
                "end_frame": prev,
                "n_frames": n,
                "duration_ps": None if frame_step_ps is None else n * frame_step_ps,
            }
        )
    return pd.DataFrame(rows)


def stats_file(input_path: str, output_path: str) -> pd.DataFrame:
    df = read_table(input_path)
    result = occupancy_by_target(df)
    write_table(result, output_path)
    return result


def multi_occupancy_file(input_path: str, output_path: str) -> pd.DataFrame:
    result = multi_occupancy(read_table(input_path))
    write_table(result, output_path)
    return result


def cooccupancy_file(input_path: str, output_path: str) -> pd.DataFrame:
    result = cooccupancy_matrix(read_table(input_path))
    write_table(result.reset_index().rename(columns={"target_label": "site"}), output_path)
    return result


def conditional_file(input_path: str, output_path: str) -> pd.DataFrame:
    result = conditional_occupancy(read_table(input_path))
    write_table(result, output_path)
    return result


def residence_file(input_path: str, output_path: str, frame_step_ps: float | None = None) -> pd.DataFrame:
    result = residence_times(read_table(input_path), frame_step_ps=frame_step_ps)
    write_table(result, output_path)
    return result
