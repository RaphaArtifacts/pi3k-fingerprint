# pi3k-fingerprint

Dynamic protein–ligand interaction fingerprinting from molecular dynamics (MD) simulation data, applied to Class I PI3K inhibitor selectivity.

This repository contains the analysis code accompanying the study of PI3K isoform-selective inhibitor binding modes based on 100 ns all-atom MD simulations of 14 inhibitor-bound complexes (PI3Kα ×4, PI3Kβ ×2, PI3Kδ ×4, PI3Kγ ×4).

## What it does

1. **Parse** standard Desmond MD analysis exports (`.dat` files: RMSD/RMSF, ligand properties, torsions, seven protein–ligand contact types).
2. **Map** residues across isoforms to a common reference numbering (human PIK3CA) via UniProt sequence alignment, enabling residue-level comparison.
3. **Analyze** contact occupancy over the equilibrated trajectory segment (fraction of frames in which each contact is present), ligand property statistics, torsion flexibility, and kinase-domain RMSF.
4. **Plot** per-isoform and cross-isoform figures: RMSD/RMSF profiles, contact-occupancy heatmaps, ligand-property bars, torsion dials, and interaction-fingerprint radar charts.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+, pandas, numpy, matplotlib, biopython, pyyaml.

## Quickstart

Run the full pipeline on two bundled demo systems (4JPS: typical hinge-anchored mode; 5T8F: atypical hinge-independent mode):

```bash
python examples/quickstart.py
```

Results (tables + figures) are written to `examples/results/`.

## Full pipeline

```bash
python -m plfingerprint run --config config/systems.yaml \
    --datadir /path/to/desmond-exports \
    --outdir results/ \
    --uniprot-json /path/to/uniprot.json
```

Individual stages can be run separately: `parse`, `map`, `analyze`, `plot`.

### Input layout

```
<datadir>/<iso>/data_<PDB>_ref/raw-data/*.dat
```

with `iso` in {alpha, beta, delta, gamma}. The `--uniprot-json` file maps sequence keys (as in `config/systems.yaml`) to full-length amino-acid sequences.

### Outputs

- `systems.pkl`: parsed dataset for all systems
- `resmap.pkl`: equivalent-residue mappings
- `contact_occupancy.csv`: long-form occupancy table (system × contact type × residue, reference numbering)
- `ligand_stats.csv`, `torsion_stats.csv`, `rmsf_kinase.csv`: summary statistics
- `figures/`: PNG figures (per-isoform + cross-isoform comparison)

## Method notes

- Occupancy of a contact = fraction of post-equilibration frames containing it. ≥50%: major interaction; 20–50%: moderate/intermittent; <5%: negligible.
- Contact residue numbers from Desmond exports are validated against UniProt sequences (native numbering, not RMSF order indices).
- Multi-system means describe cross-structure variation, not biological replicates.

## License

MIT. This repository is a static companion to a publication and is not actively maintained.

## Citation

If you use this code, please cite the accompanying paper (citation to be added upon publication).
