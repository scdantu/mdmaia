"""Command-line interface for MD-MAIA."""

from __future__ import annotations

import argparse

from .compare import compare_occupancy_file, merge_condition_tables
from .config import collect_from_config
from .collect import collect_from_args
from .density import density_file
from .export import site_pdb_file, write_pymol_density_script
from .features import features_file
from .plot import plot_cooccupancy, plot_occupancy, write_feature_matrix
from .sites import cluster_sites_file
from .stats import stats_file
from .stats import cooccupancy_file, conditional_file, multi_occupancy_file, residence_file
from .states import classify_file, state_summary_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mdmaia",
        description="MD-MAIA: Molecular Dynamics Machine-learning Assisted Interaction Analysis",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="Collect PBC-aware local ligand/cofactor positions")
    collect.add_argument("--topology", "-s", required=True, help="Topology/structure file")
    collect.add_argument("--trajectory", "-f", required=True, help="Trajectory file")
    collect.add_argument("--target", required=True, help="MDAnalysis selection for target site")
    collect.add_argument("--mobile", required=True, help="MDAnalysis selection for ligand/cofactor/ion")
    collect.add_argument("--target-label", default="target", help="Label for this target/site")
    collect.add_argument("--cutoff", type=float, default=None, help="Keep only mobile atoms within cutoff")
    collect.add_argument("--start", type=int, default=None, help="First frame")
    collect.add_argument("--stop", type=int, default=None, help="Stop frame")
    collect.add_argument("--step", type=int, default=None, help="Frame stride")
    collect.add_argument("--output", "-o", required=True, help="Output CSV or Parquet table")
    collect.set_defaults(func=collect_from_args)

    collect_config = sub.add_parser(
        "collect-config",
        help="Collect local positions for multiple targets from a YAML config",
    )
    collect_config.add_argument("--config", "-c", required=True, help="YAML config file")
    collect_config.set_defaults(func=lambda args: collect_from_config(args.config))

    density = sub.add_parser("density", help="Generate OpenDX density from collected positions")
    density.add_argument("--input", "-i", required=True, help="Input CSV/Parquet table")
    density.add_argument("--output", "-o", required=True, help="Output .dx file")
    density.add_argument("--spacing", type=float, default=0.5, help="Grid spacing")
    density.add_argument("--radius", type=float, default=None, help="Grid half-width around target")
    density.add_argument("--no-normalize", action="store_true", help="Write counts instead of probability")
    density.set_defaults(
        func=lambda args: density_file(
            args.input,
            args.output,
            spacing=args.spacing,
            radius=args.radius,
            normalize=not args.no_normalize,
        )
    )

    stats = sub.add_parser("stats", help="Calculate occupancy statistics")
    stats.add_argument("--input", "-i", required=True, help="Input CSV/Parquet table")
    stats.add_argument("--output", "-o", required=True, help="Output CSV/Parquet table")
    stats.set_defaults(func=lambda args: stats_file(args.input, args.output))

    multi = sub.add_parser("multi-occupancy", help="Probability of at least N mobile objects")
    multi.add_argument("--input", "-i", required=True, help="Input CSV/Parquet table")
    multi.add_argument("--output", "-o", required=True, help="Output CSV/Parquet table")
    multi.set_defaults(func=lambda args: multi_occupancy_file(args.input, args.output))

    coocc = sub.add_parser("cooccupancy", help="Calculate site-site co-occupancy matrix")
    coocc.add_argument("--input", "-i", required=True, help="Input CSV/Parquet table")
    coocc.add_argument("--output", "-o", required=True, help="Output CSV/Parquet table")
    coocc.set_defaults(func=lambda args: cooccupancy_file(args.input, args.output))

    cond = sub.add_parser("conditional", help="Calculate conditional site occupancy")
    cond.add_argument("--input", "-i", required=True, help="Input CSV/Parquet table")
    cond.add_argument("--output", "-o", required=True, help="Output CSV/Parquet table")
    cond.set_defaults(func=lambda args: conditional_file(args.input, args.output))

    residence = sub.add_parser("residence", help="Calculate consecutive residence runs")
    residence.add_argument("--input", "-i", required=True, help="Input CSV/Parquet table")
    residence.add_argument("--output", "-o", required=True, help="Output CSV/Parquet table")
    residence.add_argument("--frame-step-ps", type=float, default=None, help="Time per analysed frame")
    residence.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help="Frame-number stride for consecutive residence events; inferred if omitted",
    )
    residence.set_defaults(
        func=lambda args: residence_file(
            args.input,
            args.output,
            args.frame_step_ps,
            args.frame_stride,
        )
    )

    features = sub.add_parser("features", help="Generate frame-wise ML-ready features")
    features.add_argument("--input", "-i", required=True, help="Input CSV/Parquet table")
    features.add_argument("--output", "-o", required=True, help="Output CSV/Parquet table")
    features.set_defaults(func=lambda args: features_file(args.input, args.output))

    feature_matrix = sub.add_parser("feature-matrix", help="Export numeric features as .npy")
    feature_matrix.add_argument("--input", "-i", required=True, help="Input feature table")
    feature_matrix.add_argument("--output", "-o", required=True, help="Output .npy file")
    feature_matrix.set_defaults(func=lambda args: write_feature_matrix(args.input, args.output))

    sites = sub.add_parser("sites", help="Cluster local coordinates into hotspot sites")
    sites.add_argument("--input", "-i", required=True, help="Input CSV/Parquet table")
    sites.add_argument("--output", "-o", required=True, help="Output CSV/Parquet table")
    sites.add_argument("--eps", type=float, default=1.0, help="DBSCAN radius")
    sites.add_argument("--min-samples", type=int, default=20, help="DBSCAN minimum samples")
    sites.set_defaults(
        func=lambda args: cluster_sites_file(args.input, args.output, args.eps, args.min_samples)
    )

    classify = sub.add_parser("classify", help="Classify distance-based interaction states")
    classify.add_argument("--input", "-i", required=True, help="Input CSV/Parquet table")
    classify.add_argument("--output", "-o", required=True, help="Output CSV/Parquet table")
    classify.add_argument("--bound-cutoff", type=float, required=True, help="Bound-state cutoff")
    classify.add_argument(
        "--encounter-cutoff",
        type=float,
        required=True,
        help="Encounter-state cutoff",
    )
    classify.add_argument("--reactive-cutoff", type=float, default=None, help="Reactive cutoff")
    classify.set_defaults(
        func=lambda args: classify_file(
            args.input,
            args.output,
            args.bound_cutoff,
            args.encounter_cutoff,
            args.reactive_cutoff,
        )
    )

    state_summary = sub.add_parser("state-summary", help="Summarise classified state frequencies")
    state_summary.add_argument("--input", "-i", required=True, help="Classified table")
    state_summary.add_argument("--output", "-o", required=True, help="Output CSV/Parquet table")
    state_summary.set_defaults(func=lambda args: state_summary_file(args.input, args.output))

    merge = sub.add_parser("merge-conditions", help="Merge per-condition collection tables")
    merge.add_argument("--input", "-i", action="append", required=True, help="Input table")
    merge.add_argument(
        "--condition",
        action="append",
        required=True,
        help="Condition label matching each --input",
    )
    merge.add_argument("--output", "-o", required=True, help="Output merged table")
    merge.set_defaults(func=lambda args: merge_condition_tables(args.input, args.condition, args.output))

    compare = sub.add_parser("compare-occupancy", help="Compare occupancy between conditions")
    compare.add_argument("--input", "-i", required=True, help="Merged condition table")
    compare.add_argument("--output", "-o", required=True, help="Output comparison table")
    compare.add_argument("--reference", default=None, help="Reference condition")
    compare.set_defaults(
        func=lambda args: compare_occupancy_file(args.input, args.output, args.reference)
    )

    export_pymol = sub.add_parser("export-pymol", help="Write PyMOL script for density map")
    export_pymol.add_argument("--structure", required=True, help="Structure file")
    export_pymol.add_argument("--density", required=True, help="OpenDX density file")
    export_pymol.add_argument("--output", "-o", required=True, help="Output .pml file")
    export_pymol.add_argument("--level", type=float, default=0.01, help="Isomesh level")
    export_pymol.set_defaults(
        func=lambda args: write_pymol_density_script(
            args.structure,
            args.density,
            args.output,
            mesh_level=args.level,
        )
    )

    export_sites = sub.add_parser("export-sites", help="Write clustered sites as pseudoatom PDB")
    export_sites.add_argument("--input", "-i", required=True, help="Sites table")
    export_sites.add_argument("--output", "-o", required=True, help="Output PDB file")
    export_sites.set_defaults(func=lambda args: site_pdb_file(args.input, args.output))

    plot_occ = sub.add_parser("plot-occupancy", help="Plot occupancy bar graph")
    plot_occ.add_argument("--input", "-i", required=True, help="Occupancy table")
    plot_occ.add_argument("--output", "-o", required=True, help="Output image")
    plot_occ.set_defaults(func=lambda args: plot_occupancy(args.input, args.output))

    plot_coocc = sub.add_parser("plot-cooccupancy", help="Plot co-occupancy heatmap")
    plot_coocc.add_argument("--input", "-i", required=True, help="Co-occupancy table")
    plot_coocc.add_argument("--output", "-o", required=True, help="Output image")
    plot_coocc.set_defaults(func=lambda args: plot_cooccupancy(args.input, args.output))

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
