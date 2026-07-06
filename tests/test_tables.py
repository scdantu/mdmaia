import pandas as pd

from mdmaia.features import frame_features
from mdmaia.sites import cluster_sites
from mdmaia.states import classify_distance_states, state_summary
from mdmaia.stats import cooccupancy_matrix, residence_times


def toy_table():
    return pd.DataFrame(
        [
            {"frame": 0, "time_ps": 0.0, "target_label": "A", "mobile_index": 1, "x": 0, "y": 0, "z": 0, "distance": 1.0},
            {"frame": 1, "time_ps": 1.0, "target_label": "A", "mobile_index": 1, "x": 1, "y": 0, "z": 0, "distance": 1.5},
            {"frame": 1, "time_ps": 1.0, "target_label": "B", "mobile_index": 2, "x": 5, "y": 0, "z": 0, "distance": 2.0},
            {"frame": 3, "time_ps": 3.0, "target_label": "A", "mobile_index": 1, "x": 1, "y": 1, "z": 0, "distance": 3.5},
        ]
    )


def test_cooccupancy_matrix():
    mat = cooccupancy_matrix(toy_table())
    assert mat.loc["A", "A"] == 1.0
    assert mat.loc["A", "B"] == 1 / 3


def test_residence_times_splits_gaps():
    res = residence_times(toy_table(), frame_step_ps=10.0)
    a = res[(res["target_label"] == "A") & (res["mobile_index"] == 1)]
    assert list(a["n_frames"]) == [2, 1]
    assert list(a["duration_ps"]) == [20.0, 10.0]


def test_frame_features():
    feat = frame_features(toy_table())
    assert set(["frame", "n_mobile", "min_distance"]).issubset(feat.columns)
    assert int(feat.loc[feat["frame"] == 1, "n_mobile"].iloc[0]) == 2


def test_state_classification_and_summary():
    classified = classify_distance_states(toy_table(), bound_cutoff=2.0, encounter_cutoff=4.0)
    assert set(classified["state"]) == {"bound", "encounter"}
    summary = state_summary(classified)
    assert not summary.empty


def test_site_clustering():
    df = pd.DataFrame(
        [
            {"x": 0.0, "y": 0.0, "z": 0.0},
            {"x": 0.1, "y": 0.0, "z": 0.0},
            {"x": 5.0, "y": 5.0, "z": 5.0},
            {"x": 5.1, "y": 5.0, "z": 5.0},
        ]
    )
    sites = cluster_sites(df, eps=0.3, min_samples=2)
    assert len(sites) == 2
