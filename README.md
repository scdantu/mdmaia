# MD-MAIA

**Molecular Dynamics – Machine-learning Assisted Interaction Analysis**

MD-MAIA is a Python toolkit for analysing ligand, ion and cofactor behaviour in
molecular dynamics simulations.

The current focus is target-centred analysis: for every trajectory frame,
MD-MAIA back-calculates the nearest periodic image of a molecule of interest
around a target site. This preserves the main advantage of the original LDAT
prototype: local ligand/cofactor positions can be reconstructed without first
writing a globally PBC-corrected trajectory.

## Current capabilities

- PBC-aware local coordinate collection around a target site.
- YAML-driven collection around multiple target sites.
- Works with arbitrary MDAnalysis selections, for example ions, waters,
  ligands or cofactors.
- CSV/Parquet output for downstream analysis.
- OpenDX density-map generation from collected local coordinates.
- Occupancy summaries from frame-wise local-coordinate tables.
- Multi-occupancy, site co-occupancy and conditional occupancy.
- Residence-time/run-length analysis.
- Distance-based state classification.
- Condition-level occupancy comparison.
- Simple frame-wise ML-ready feature extraction.
- Hotspot/site clustering utilities.
- PyMOL density-map script export and hotspot pseudoatom PDB export.

## Install for development

```bash
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[parquet,plot,dev]"
```

## Example usage

Collect Mn positions around a catalytic site:

```bash
mdmaia collect \
  --topology topol.tpr \
  --trajectory traj.xtc \
  --target "nucleic and resid 22 and name P" \
  --mobile "resname MN" \
  --target-label G22 \
  --cutoff 8.0 \
  --output mn_g22.csv
```

Generate a density map:

```bash
mdmaia density \
  --input mn_g22.csv \
  --spacing 0.5 \
  --radius 8.0 \
  --output mn_g22.dx
```

Calculate occupancy statistics:

```bash
mdmaia stats \
  --input mn_g22.csv \
  --output mn_g22_occupancy.csv
```

Generate simple frame-wise features:

```bash
mdmaia features \
  --input mn_g22.csv \
  --output mn_g22_features.csv
```

Export a numeric feature matrix for ML:

```bash
mdmaia feature-matrix \
  --input mn_g22_features.csv \
  --output mn_g22_features.npy
```

## Multi-site workflow

Use a YAML file for repeated analyses around several target sites:

```yaml
topology: topol.tpr
trajectory: traj.xtc
mobile: "resname MN"
output: mn_sites.csv
cutoff: 8.0
targets:
  - label: G22
    selection: "nucleic and resid 22 and name P"
  - label: patch_61_66
    selection: "nucleic and resid 61-66 and name P"
  - label: site_75_77
    selection: "nucleic and resid 75-77 and name P"
```

Run:

```bash
mdmaia collect-config --config examples/mn_dna.yaml
```

Then analyse co-occupancy and residence:

```bash
mdmaia cooccupancy \
  --input mn_sites.csv \
  --output mn_sites_cooccupancy.csv

mdmaia conditional \
  --input mn_sites.csv \
  --output mn_sites_conditional.csv

mdmaia residence \
  --input mn_sites.csv \
  --frame-step-ps 20 \
  --output mn_sites_residence.csv
```

Classify simple interaction states:

```bash
mdmaia classify \
  --input mn_sites.csv \
  --bound-cutoff 3.5 \
  --encounter-cutoff 8.0 \
  --output mn_sites_states.csv

mdmaia state-summary \
  --input mn_sites_states.csv \
  --output mn_sites_state_summary.csv
```

Detect density hotspots:

```bash
mdmaia sites \
  --input mn_sites.csv \
  --eps 1.0 \
  --min-samples 20 \
  --output mn_hotspots.csv

mdmaia export-sites \
  --input mn_hotspots.csv \
  --output mn_hotspots.pdb
```

Generate PyMOL helper script:

```bash
mdmaia export-pymol \
  --structure reference.pdb \
  --density mn_g22.dx \
  --level 0.01 \
  --output view_mn_density.pml
```

## Design direction

MD-MAIA should grow into two linked layers:

1. **Physical/statistical analysis**
   - local density maps
   - site occupancy
   - multi-site co-occupancy
   - residence/contact duration
   - condition and replica comparisons

2. **AI/ML-ready featurisation**
   - frame-wise interaction fingerprints
   - distance/contact matrices
   - site-state labels
   - density-derived features
   - graph representations for ligand/cofactor/residue networks

The old LDAT prototype remains in the repository for reference but new
development should target the `src/mdmaia` package.

## Implemented CLI commands

```text
mdmaia collect
mdmaia collect-config
mdmaia density
mdmaia stats
mdmaia multi-occupancy
mdmaia cooccupancy
mdmaia conditional
mdmaia residence
mdmaia features
mdmaia feature-matrix
mdmaia sites
mdmaia classify
mdmaia state-summary
mdmaia merge-conditions
mdmaia compare-occupancy
mdmaia export-pymol
mdmaia export-sites
mdmaia plot-occupancy
mdmaia plot-cooccupancy
```
