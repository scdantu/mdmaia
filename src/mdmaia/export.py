"""Export helpers for molecular viewers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io import read_table


def write_pymol_density_script(
    structure: str,
    density: str,
    output: str,
    mesh_level: float = 0.01,
    object_name: str = "mdmaia_density",
) -> None:
    """Write a simple PyMOL script for viewing a density map."""

    text = f"""\
reinitialize
bg_color white
load {structure}, structure
load {density}, {object_name}
hide everything
show cartoon, structure
isomesh {object_name}_mesh, {object_name}, {mesh_level}
color marine, structure
color tv_orange, {object_name}_mesh
set mesh_width, 0.5
orient all
zoom all, 5
"""
    Path(output).write_text(text)


def write_site_pdb(sites: pd.DataFrame, output: str, resname: str = "SIT") -> None:
    """Write density/site centres as pseudo-atoms in a PDB file."""

    lines = []
    for i, row in sites.reset_index(drop=True).iterrows():
        lines.append(
            "HETATM{atom_id:5d}  X   {resname:>3s} A{resid:4d}    "
            "{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}           X\n".format(
                atom_id=i + 1,
                resname=resname,
                resid=i + 1,
                x=float(row["x"]),
                y=float(row["y"]),
                z=float(row["z"]),
                b=float(row.get("fraction", row.get("n_points", 0.0))),
            )
        )
    lines.append("END\n")
    Path(output).write_text("".join(lines))


def site_pdb_file(input_path: str, output_path: str) -> None:
    write_site_pdb(read_table(input_path), output_path)
