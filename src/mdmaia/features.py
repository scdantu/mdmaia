"""ML-ready feature extraction from local-coordinate tables."""

from __future__ import annotations

import pandas as pd

from .io import read_table, write_table


def frame_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert local-coordinate records into one row per trajectory frame."""

    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby("frame", observed=True)
    features = grouped.agg(
        time_ps=("time_ps", "first"),
        n_mobile=("mobile_index", "nunique"),
        min_distance=("distance", "min"),
        mean_distance=("distance", "mean"),
        max_distance=("distance", "max"),
        mean_x=("x", "mean"),
        mean_y=("y", "mean"),
        mean_z=("z", "mean"),
    )
    return features.reset_index()


def features_file(input_path: str, output_path: str) -> pd.DataFrame:
    df = read_table(input_path)
    result = frame_features(df)
    write_table(result, output_path)
    return result
