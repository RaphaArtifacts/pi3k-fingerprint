"""Parse Desmond molecular dynamics analysis exports (.dat files).

Each simulated system is expected to provide the standard Desmond analysis
exports inside ``<raw-data>/``:
``PL_RMSD.dat``, ``P_RMSF.dat``, ``L_RMSF.dat``, ``L-Properties.dat``,
``L_Torsions.dat`` and ``PL-Contacts_<type>.dat`` for each contact type.
"""
import os
import re
import pickle

import numpy as np
import pandas as pd

DEFAULT_CONTACT_TYPES = ["HBond", "Hydrophobic", "WaterBridge", "Ionic",
                         "Pi-Pi", "Pi-Cation", "Metal"]


def raw_dir(datadir, iso, pdb, dir_exceptions=None):
    """Return the raw-data directory for one system."""
    d = (dir_exceptions or {}).get(pdb, f"data_{pdb}_ref")
    return os.path.join(datadir, iso, d, "raw-data")


def read_table(path, comment=("#",)):
    """Read a whitespace-delimited .dat table, skipping comments/headers."""
    rows = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith(comment) or s.startswith("Frame"):
                continue
            rows.append(s.split())
    return rows


def parse_system(datadir, iso, pdb, dir_exceptions=None,
                 contact_types=None):
    """Parse all .dat exports for a single system.

    Returns a dict with keys: ``rmsd``, ``p_rmsf``, ``l_rmsf``, ``l_props``,
    ``torsions`` (DataFrames), ``contacts`` (dict of DataFrames per type),
    ``occupancy`` (dict of per-residue occupancy Series per type),
    ``n_frames``, ``iso`` and ``pdb``.
    """
    contact_types = contact_types or DEFAULT_CONTACT_TYPES
    rd = raw_dir(datadir, iso, pdb, dir_exceptions)
    sys = {"iso": iso, "pdb": pdb}

    # --- RMSD ---
    rows = read_table(os.path.join(rd, "PL_RMSD.dat"))
    cols = ["frame", "Prot_CA", "Prot_Backbone", "Prot_Sidechain",
            "Prot_All_Heavy", "Lig_wrt_Protein", "Lig_wrt_Ligand"]
    rmsd = pd.DataFrame([[float(x) for x in r[:7]] for r in rows],
                        columns=cols)
    rmsd["time_ns"] = rmsd["frame"] * 0.1
    sys["rmsd"] = rmsd
    sys["n_frames"] = len(rmsd)

    # --- protein RMSF ---
    rows = read_table(os.path.join(rd, "P_RMSF.dat"))
    recs = []
    for r in rows:
        # idx, chain, ResName_LigandContact..., CA, Backbone, Sidechain, All_Heavy
        m = re.match(r"([A-Z]+)_(\d+)", r[2])
        recs.append(dict(idx=int(r[0]), chain=r[1], resname=m.group(1),
                         resnum=int(m.group(2)), contact=r[3],
                         CA=float(r[4]), Backbone=float(r[5]),
                         Sidechain=float(r[6]), All_Heavy=float(r[7])))
    sys["p_rmsf"] = pd.DataFrame(recs)

    # --- ligand RMSF ---
    rows = read_table(os.path.join(rd, "L_RMSF.dat"))
    sys["l_rmsf"] = pd.DataFrame(
        [dict(atom=int(r[0]), wrt_Protein=float(r[1]),
              wrt_Ligand=float(r[2])) for r in rows])

    # --- ligand properties ---
    rows = read_table(os.path.join(rd, "L-Properties.dat"))
    sys["l_props"] = pd.DataFrame(
        [[float(x) for x in r[:7]] for r in rows],
        columns=["frame", "RMSD", "rGyr", "intraHB", "MolSA", "SASA", "PSA"])

    # --- torsions ---
    rows = read_table(os.path.join(rd, "L_Torsions.dat"))
    ncol = len(rows[0]) - 1
    sys["torsions"] = pd.DataFrame(
        [[float(x) for x in r[1:1 + ncol]] for r in rows],
        columns=[f"dihed{i + 1}" for i in range(ncol)])

    # --- contacts ---
    contacts, occup = {}, {}
    n_frames = len(rmsd)
    for ct in contact_types:
        fn = os.path.join(rd, f"PL-Contacts_{ct}.dat")
        rows = read_table(fn)
        if ct == "Metal" or not rows:
            contacts[ct] = pd.DataFrame(columns=["frame"])
            occup[ct] = pd.Series(dtype=float)
            continue
        recs = []
        for r in rows:
            # frame, Residue#, Chain, ResName, ...
            recs.append(dict(frame=int(r[0]), idx=int(r[1]),
                             chain=r[2], resname=r[3]))
        df = pd.DataFrame(recs)
        contacts[ct] = df
        occup[ct] = df.groupby("idx")["frame"].nunique() / n_frames
    sys["contacts"] = contacts
    sys["occupancy"] = occup
    return sys


def parse_all(datadir, systems, dir_exceptions=None, contact_types=None,
              out_pkl=None):
    """Parse every system in ``systems`` (dict iso -> [pdb, ...]).

    Returns ``{iso_pdb: system_dict}`` and optionally pickles it to
    ``out_pkl``.
    """
    data = {}
    for iso, pdbs in systems.items():
        for pdb in pdbs:
            key = f"{iso}_{pdb}"
            print(f"parsing {key}", flush=True)
            data[key] = parse_system(datadir, iso, pdb, dir_exceptions,
                                     contact_types)
    if out_pkl:
        with open(out_pkl, "wb") as fh:
            pickle.dump(data, fh)
    return data
