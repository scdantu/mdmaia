"""Condition/replica comparison helpers."""

from __future__ import annotations

import pandas as pd

from .io import read_table, write_table
from .stats import occupancy_by_target


def add_condition(df: pd.DataFrame, condition: str, replica: str | None = None) -> pd.DataFrame:
    out = df.copy()
    out["condition"] = condition
    if replica is not None:
        out["replica"] = replica
    return out


def merge_condition_tables(inputs: list[str], conditions: list[str], output_path: str) -> pd.DataFrame:
    if len(inputs) != len(conditions):
        raise ValueError("Number of inputs must match number of conditions.")
    frames = [add_condition(read_table(path), condition) for path, condition in zip(inputs, conditions)]
    result = pd.concat(frames, ignore_index=True)
    write_table(result, output_path)
    return result


def occupancy_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    if "condition" not in df.columns:
        raise ValueError("Input table must contain a 'condition' column.")
    rows = []
    for condition, group in df.groupby("condition", observed=True):
        occ = occupancy_by_target(group)
        occ.insert(0, "condition", condition)
        rows.append(occ)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compare_occupancy(df: pd.DataFrame, reference: str | None = None) -> pd.DataFrame:
    occ = occupancy_by_condition(df)
    if occ.empty:
        return occ
    wide = occ.pivot_table(
        index="target_label",
        columns="condition",
        values="occupancy",
        observed=True,
    ).reset_index()
    conditions = [c for c in wide.columns if c != "target_label"]
    if reference is None:
        reference = conditions[0]
    for condition in conditions:
        if condition == reference:
            continue
        wide[f"delta_{condition}_minus_{reference}"] = wide[condition] - wide[reference]
    return wide


def compare_occupancy_file(
    input_path: str,
    output_path: str,
    reference: str | None = None,
) -> pd.DataFrame:
    result = compare_occupancy(read_table(input_path), reference=reference)
    write_table(result, output_path)
    return result
