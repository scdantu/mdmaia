"""Occupancy, co-occupancy and residence statistics."""

from __future__ import annotations

import pandas as pd

from .io import read_table, write_table


def _metadata_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in ["condition", "replica"] if col in df.columns and df[col].notna().any()]


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

    meta = _metadata_columns(df)
    frame_counts = (
        df.groupby(meta + ["target_label", "frame"], observed=True)["mobile_index"]
        .nunique()
        .reset_index(name="mobile_count")
    )
    total_frames = (
        df.groupby(meta, observed=True)["frame"].nunique().rename("n_frames_total").reset_index()
        if meta
        else pd.DataFrame({"n_frames_total": [df["frame"].nunique()]})
    )
    out = (
        frame_counts.groupby(meta + ["target_label"], observed=True)
        .agg(
            n_frames_occupied=("frame", "nunique"),
            mean_mobile_count=("mobile_count", "mean"),
            max_mobile_count=("mobile_count", "max"),
        )
        .reset_index()
    )
    if meta:
        out = out.merge(total_frames, on=meta, how="left")
    else:
        out["n_frames_total"] = int(total_frames["n_frames_total"].iloc[0])
    out["occupancy"] = out["n_frames_occupied"] / out["n_frames_total"]
    ordered = meta + [
        "target_label",
        "n_frames_total",
        "n_frames_occupied",
        "occupancy",
        "mean_mobile_count",
        "max_mobile_count",
    ]
    return out[ordered]


def multi_occupancy(df: pd.DataFrame) -> pd.DataFrame:
    """Probability of observing at least N mobile atoms per frame."""

    if df.empty:
        return pd.DataFrame(columns=["threshold", "n_frames", "probability"])
    rows = []
    meta = _metadata_columns(df)
    grouped = df.groupby(meta, observed=True) if meta else [((), df)]
    for key, group in grouped:
        counts = group.groupby("frame")["mobile_index"].nunique()
        n_frames = counts.shape[0]
        key_values = key if isinstance(key, tuple) else (key,)
        meta_data = dict(zip(meta, key_values))
        for threshold in range(1, int(counts.max()) + 1):
            hits = int((counts >= threshold).sum())
            rows.append({**meta_data, "threshold": threshold, "n_frames": hits, "probability": hits / n_frames})
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


def cooccupancy_long(df: pd.DataFrame) -> pd.DataFrame:
    """Long-format co-occupancy, grouped by condition/replica when present."""

    if df.empty:
        return pd.DataFrame(columns=["site_a", "site_b", "cooccupancy"])
    meta = _metadata_columns(df)
    rows = []
    grouped = df.groupby(meta, observed=True) if meta else [((), df)]
    for key, group in grouped:
        matrix = cooccupancy_matrix(group)
        key_values = key if isinstance(key, tuple) else (key,)
        meta_data = dict(zip(meta, key_values))
        for site_a in matrix.index:
            for site_b in matrix.columns:
                rows.append(
                    {
                        **meta_data,
                        "site_a": site_a,
                        "site_b": site_b,
                        "cooccupancy": float(matrix.loc[site_a, site_b]),
                    }
                )
    return pd.DataFrame(rows)


def conditional_occupancy(df: pd.DataFrame) -> pd.DataFrame:
    """Conditional probability ``P(site_b occupied | site_a occupied)``."""

    if df.empty:
        return pd.DataFrame(columns=["site_a", "site_b", "p_b_given_a"])
    rows = []
    meta = _metadata_columns(df)
    grouped = df.groupby(meta, observed=True) if meta else [((), df)]
    for key, group in grouped:
        presence = (
            group.assign(occupied=1)
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
        key_values = key if isinstance(key, tuple) else (key,)
        meta_data = dict(zip(meta, key_values))
        for site_a in presence.columns:
            denom = presence[site_a].sum()
            for site_b in presence.columns:
                prob = (
                    0.0
                    if denom == 0
                    else float(((presence[site_a] == 1) & (presence[site_b] == 1)).sum() / denom)
                )
                rows.append({**meta_data, "site_a": site_a, "site_b": site_b, "p_b_given_a": prob})
    return pd.DataFrame(rows)


def residence_times(
    df: pd.DataFrame,
    frame_step_ps: float | None = None,
    frame_stride: int | None = None,
    by_mobile: bool = True,
) -> pd.DataFrame:
    """Consecutive occupied-frame runs.

    By default, runs are ion-specific: ``target_label`` + ``mobile_index``.
    With ``by_mobile=False``, runs are site-level: a site is occupied if any
    mobile object occupies it in that frame.
    """

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

    if frame_stride is None:
        frame_stride = _infer_frame_stride(df["frame"])

    meta = _metadata_columns(df)
    group_cols = meta + ["target_label"] + (["mobile_index"] if by_mobile else [])
    rows = []
    for key, group in df.groupby(group_cols, observed=True):
        key_values = key if isinstance(key, tuple) else (key,)
        group_data = dict(zip(group_cols, key_values))
        frames = sorted(set(int(x) for x in group["frame"]))
        if not frames:
            continue
        start = prev = frames[0]
        n_observed = 1
        for frame in frames[1:]:
            if frame == prev + frame_stride:
                prev = frame
                n_observed += 1
                continue
            row = {
                "target_label": group_data["target_label"],
                "mobile_index": group_data.get("mobile_index"),
                "start_frame": start,
                "end_frame": prev,
                "n_frames": n_observed,
                "duration_ps": None if frame_step_ps is None else n_observed * frame_step_ps,
            }
            row.update({col: group_data[col] for col in meta})
            rows.append(row)
            start = prev = frame
            n_observed = 1
        row = {
            "target_label": group_data["target_label"],
            "mobile_index": group_data.get("mobile_index"),
            "start_frame": start,
            "end_frame": prev,
            "n_frames": n_observed,
            "duration_ps": None if frame_step_ps is None else n_observed * frame_step_ps,
        }
        row.update({col: group_data[col] for col in meta})
        rows.append(row)
    return pd.DataFrame(rows)


def site_residence_times(
    df: pd.DataFrame,
    frame_step_ps: float | None = None,
    frame_stride: int | None = None,
) -> pd.DataFrame:
    """Site-level residence: occupied by any equivalent mobile object."""

    return residence_times(
        df,
        frame_step_ps=frame_step_ps,
        frame_stride=frame_stride,
        by_mobile=False,
    )


def cluster_residence_summary(
    df: pd.DataFrame,
    frame_step_ps: float | None = None,
    frame_stride: int | None = None,
) -> pd.DataFrame:
    """Occupancy and residence-time summary for assigned contact clusters.

    Expects a ``cluster_id`` column. Cluster ``-1`` is treated as noise and
    excluded from the summary.
    """

    if df.empty or "cluster_id" not in df.columns:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "n_points",
                "n_frames_occupied",
                "occupancy",
                "mean_mobile_count",
                "max_mobile_count",
                "n_residence_events",
                "mean_residence_ps",
                "max_residence_ps",
                "dominant_target_label",
            ]
        )

    clustered = df[df["cluster_id"] >= 0].copy()
    if clustered.empty:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "n_points",
                "n_frames_occupied",
                "occupancy",
                "mean_mobile_count",
                "max_mobile_count",
                "n_residence_events",
                "mean_residence_ps",
                "max_residence_ps",
                "dominant_target_label",
            ]
        )

    meta = _metadata_columns(clustered)
    total_frames = (
        df.groupby(meta, observed=True)["frame"].nunique().rename("n_frames_total").reset_index()
        if meta
        else pd.DataFrame({"n_frames_total": [df["frame"].nunique()]})
    )
    frame_counts = (
        clustered.groupby(meta + ["cluster_id", "frame"], observed=True)["mobile_index"]
        .nunique()
        .reset_index(name="mobile_count")
    )
    occ = (
        frame_counts.groupby(meta + ["cluster_id"], observed=True)
        .agg(
            n_frames_occupied=("frame", "nunique"),
            mean_mobile_count=("mobile_count", "mean"),
            max_mobile_count=("mobile_count", "max"),
        )
        .reset_index()
    )
    if meta:
        occ = occ.merge(total_frames, on=meta, how="left")
    else:
        occ["n_frames_total"] = int(total_frames["n_frames_total"].iloc[0])
    occ["occupancy"] = occ["n_frames_occupied"] / occ["n_frames_total"]

    point_counts = clustered.groupby(meta + ["cluster_id"], observed=True).size().rename("n_points")
    dominant_targets = (
        clustered.groupby(meta + ["cluster_id", "target_label"], observed=True)
        .size()
        .reset_index(name="n")
        .sort_values(meta + ["cluster_id", "n"], ascending=[True] * (len(meta) + 1) + [False])
        .drop_duplicates(meta + ["cluster_id"])
        [meta + ["cluster_id", "target_label"]]
        .rename(columns={"target_label": "dominant_target_label"})
    )

    residence_input = clustered.copy()
    residence_input["target_label"] = residence_input["cluster_id"].astype(str)
    runs = site_residence_times(
        residence_input,
        frame_step_ps=frame_step_ps,
        frame_stride=frame_stride,
    )
    if runs.empty:
        res_summary = pd.DataFrame(
            columns=["cluster_id", "n_residence_events", "mean_residence_ps", "max_residence_ps"]
        )
    else:
        runs["cluster_id"] = runs["target_label"].astype(int)
        res_summary = (
            runs.groupby(meta + ["cluster_id"], observed=True)
            .agg(
                n_residence_events=("n_frames", "size"),
                mean_residence_ps=("duration_ps", "mean"),
                max_residence_ps=("duration_ps", "max"),
            )
            .reset_index()
        )

    result = (
        occ.merge(point_counts.reset_index(), on=meta + ["cluster_id"], how="left")
        .merge(res_summary, on=meta + ["cluster_id"], how="left")
        .merge(dominant_targets, on=meta + ["cluster_id"], how="left")
    )
    ordered = meta + [
        "cluster_id",
        "n_points",
        "n_frames_total",
        "n_frames_occupied",
        "occupancy",
        "mean_mobile_count",
        "max_mobile_count",
        "n_residence_events",
        "mean_residence_ps",
        "max_residence_ps",
        "dominant_target_label",
    ]
    return result[ordered].sort_values(meta + ["occupancy", "n_points"], ascending=[True] * len(meta) + [False, False])


def _infer_frame_stride(frames: pd.Series) -> int:
    unique_frames = sorted(set(int(x) for x in frames))
    if len(unique_frames) < 2:
        return 1
    diffs = [b - a for a, b in zip(unique_frames[:-1], unique_frames[1:]) if b > a]
    return min(diffs) if diffs else 1


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
    df = read_table(input_path)
    if _metadata_columns(df):
        result = cooccupancy_long(df)
        write_table(result, output_path)
    else:
        result = cooccupancy_matrix(df)
        write_table(result.reset_index().rename(columns={"target_label": "site"}), output_path)
    return result


def conditional_file(input_path: str, output_path: str) -> pd.DataFrame:
    result = conditional_occupancy(read_table(input_path))
    write_table(result, output_path)
    return result


def residence_file(
    input_path: str,
    output_path: str,
    frame_step_ps: float | None = None,
    frame_stride: int | None = None,
    by_mobile: bool = True,
) -> pd.DataFrame:
    result = residence_times(
        read_table(input_path),
        frame_step_ps=frame_step_ps,
        frame_stride=frame_stride,
        by_mobile=by_mobile,
    )
    write_table(result, output_path)
    return result


def cluster_residence_file(
    input_path: str,
    output_path: str,
    frame_step_ps: float | None = None,
    frame_stride: int | None = None,
) -> pd.DataFrame:
    result = cluster_residence_summary(
        read_table(input_path),
        frame_step_ps=frame_step_ps,
        frame_stride=frame_stride,
    )
    write_table(result, output_path)
    return result
