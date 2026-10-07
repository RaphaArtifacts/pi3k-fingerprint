"""Command-line interface for plfingerprint.

Pipeline stages:
    parse      Parse Desmond .dat exports into a structured dataset.
    map        Build equivalent-residue mappings (UniProt sequences).
    analyze    Compute contact occupancy and summary statistics.
    plot       Generate per-isoform and cross-isoform figures.
    run        Execute all stages in order.

Example:
    python -m plfingerprint run --config config/systems.yaml \\
        --datadir /path/to/dat --outdir results/
"""
import argparse
import os
import pickle
import sys

import yaml

from . import parse as parse_mod
from . import mapping as mapping_mod
from . import occupancy as occupancy_mod
from . import plots as plots_mod

ISO_COLORS = {"alpha": "#C0392B", "beta": "#2471A3",
              "delta": "#229954", "gamma": "#8E44AD"}


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def cmd_parse(args, cfg):
    os.makedirs(args.outdir, exist_ok=True)
    parse_mod.parse_all(
        args.datadir, cfg["systems"],
        dir_exceptions=cfg.get("dir_exceptions"),
        contact_types=cfg.get("contact_types"),
        out_pkl=os.path.join(args.outdir, "systems.pkl"))


def cmd_map(args, cfg):
    os.makedirs(args.outdir, exist_ok=True)
    seqs = mapping_mod.load_sequences(args.uniprot_json)
    ref = cfg["uniprot"]["reference"]
    queries = [v for k, v in cfg["uniprot"].items()
               if k not in ("reference", cfg.get("reference_iso", "alpha"))]
    maps = mapping_mod.map_uniprot_sequences(seqs, ref, queries)
    mapping_mod.save_maps(maps, os.path.join(args.outdir, "resmap.pkl"))


def cmd_analyze(args, cfg):
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "systems.pkl"), "rb") as fh:
        data = pickle.load(fh)
    resmap = mapping_mod.load_maps(os.path.join(args.outdir, "resmap.pkl"))
    seqs = mapping_mod.load_sequences(args.uniprot_json)
    systems = cfg["systems"]
    ligands = cfg.get("ligands", {})
    iso_seq_keys = {k: v for k, v in cfg["uniprot"].items()
                    if k != "reference"}
    iso_uniprot_keys = dict(iso_seq_keys)
    eq_frames = cfg.get("equilibration_frames", 200)
    ref_iso = cfg.get("reference_iso", "alpha")
    kd = tuple(cfg.get("kinase_domain_range", [700, 1068]))

    contacts, _ = occupancy_mod.contact_occupancy_table(
        data, systems, resmap, iso_seq_keys, seqs, iso_uniprot_keys,
        ligands=ligands, reference_iso=ref_iso)
    contacts.to_csv(os.path.join(args.outdir, "contact_occupancy.csv"),
                    index=False)

    ligstats = occupancy_mod.ligand_property_stats(
        data, systems, ligands=ligands, equilibration_frames=eq_frames)
    ligstats.to_csv(os.path.join(args.outdir, "ligand_stats.csv"),
                    index=False)

    tors = occupancy_mod.torsion_flexibility_stats(
        data, systems, ligands=ligands, equilibration_frames=eq_frames)
    tors.to_csv(os.path.join(args.outdir, "torsion_stats.csv"), index=False)

    rmsf = occupancy_mod.kinase_rmsf_table(
        data, systems, resmap, iso_seq_keys, reference_iso=ref_iso,
        ref_range=kd)
    rmsf.to_csv(os.path.join(args.outdir, "rmsf_kinase.csv"), index=False)
    print("analysis tables written.", flush=True)


def cmd_plot(args, cfg):
    import pandas as pd
    figdir = os.path.join(args.outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    with open(os.path.join(args.outdir, "systems.pkl"), "rb") as fh:
        data = pickle.load(fh)
    contacts = pd.read_csv(os.path.join(args.outdir,
                                        "contact_occupancy.csv"))
    ligstats = pd.read_csv(os.path.join(args.outdir, "ligand_stats.csv"))
    rmsf = pd.read_csv(os.path.join(args.outdir, "rmsf_kinase.csv"))
    systems = cfg["systems"]
    ligands = cfg.get("ligands", {})
    isos = list(systems.keys())
    iso_names = {iso: iso for iso in isos}
    eq_frames = cfg.get("equilibration_frames", 200)

    for iso, pdbs in systems.items():
        plots_mod.rmsd_figure(data, iso, pdbs, ligands, figdir)
        plots_mod.rmsf_figure(rmsf, iso, pdbs, ligands, figdir)
        plots_mod.occupancy_heatmap(contacts, iso, pdbs, figdir)
        plots_mod.ligand_property_bars(ligstats, iso, pdbs, ligands,
                                       figdir)
        plots_mod.torsion_dials(data, iso, pdbs, ligands, figdir,
                                equilibration_frames=eq_frames)
        plots_mod.fingerprint_radar(contacts, iso, pdbs, ligands, figdir)
        print(f"figures done: {iso}", flush=True)

    plots_mod.compare_rmsf(rmsf, isos, iso_names, ISO_COLORS, figdir)
    plots_mod.compare_occupancy_heatmap(contacts, systems, isos,
                                        iso_names, ISO_COLORS, figdir)
    plots_mod.compare_ligprop_radar(ligstats, isos, iso_names,
                                    ISO_COLORS, figdir)
    plots_mod.compare_fingerprint_radar(contacts, systems, isos,
                                        iso_names, ISO_COLORS, figdir)
    print("comparison figures done.", flush=True)


def build_parser():
    p = argparse.ArgumentParser(
        prog="plfingerprint",
        description="Dynamic protein-ligand interaction fingerprinting "
                    "from Desmond MD analysis exports.")
    p.add_argument("stage", choices=["parse", "map", "analyze", "plot",
                                    "run"],
                   help="pipeline stage to execute")
    p.add_argument("--config", default="config/systems.yaml",
                   help="YAML system configuration")
    p.add_argument("--datadir", default="data",
                   help="directory with per-system raw .dat exports")
    p.add_argument("--outdir", default="results",
                   help="output directory for tables and figures")
    p.add_argument("--uniprot-json", default=None,
                   help="JSON file with UniProt sequences "
                        "(required for 'map' stage)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.stage in ("parse", "run"):
        cmd_parse(args, cfg)
    if args.stage in ("map", "run"):
        if not args.uniprot_json:
            sys.exit("error: --uniprot-json is required for the "
                     "'map' stage")
        cmd_map(args, cfg)
    if args.stage in ("analyze", "run"):
        cmd_analyze(args, cfg)
    if args.stage in ("plot", "run"):
        cmd_plot(args, cfg)


if __name__ == "__main__":
    main()
