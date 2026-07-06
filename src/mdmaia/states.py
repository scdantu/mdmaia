"""State classification for ligand/cofactor local-coordinate tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import read_table, write_table


def classify_distance_states(
    df: pd.DataFrame,
    bound_cutoff: float,
    encounter_cutoff: float,
    reactive_cutoff: float | None = None,
) -> pd.DataFrame:
    """Assign simple distance-based states to each local-coordinate record."""

    if encounter_cutoff < bound_cutoff:
        raise ValueError("encounter_cutoff must be >= bound_cutoff")
    if reactive_cutoff is not None and reactive_cutoff > bound_cutoff:
        raise ValueError("reactive_cutoff must be <= bound_cutoff")

    out = df.copy()
    state = np.full(len(out), "unbound", dtype=object)
    state[out["distance"].to_numpy() <= encounter_cutoff] = "encounter"
    state[out["distance"].to_numpy() <= bound_cutoff] = "bound"
    if reactive_cutoff is not None:
        state[out["distance"].to_numpy() <= reactive_cutoff] = "reactive"
    out["state"] = state
    return out


def state_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise state frequencies per target."""

    if df.empty or "state" not in df.columns:
        return pd.DataFrame(columns=["target_label", "state", "n_records", "fraction"])
    counts = (
        df.groupby(["target_label", "state"], observed=True)
        .size()
        .reset_index(name="n_records")
    )
    totals = counts.groupby("target_label", observed=True)["n_records"].transform("sum")
    counts["fraction"] = counts["n_records"] / totals
    return counts.sort_values(["target_label", "fraction"], ascending=[True, False])


def classify_file(
    input_path: str,
    output_path: str,
    bound_cutoff: float,
    encounter_cutoff: float,
    reactive_cutoff: float | None = None,
) -> pd.DataFrame:
    result = classify_distance_states(
        read_table(input_path),
        bound_cutoff=bound_cutoff,
        encounter_cutoff=encounter_cutoff,
        reactive_cutoff=reactive_cutoff,
    )
    write_table(result, output_path)
    return result


def state_summary_file(input_path: str, output_path: str) -> pd.DataFrame:
    result = state_summary(read_table(input_path))
    write_table(result, output_path)
    return result
