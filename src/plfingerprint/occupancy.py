"""Contact occupancy statistics over the equilibrated trajectory segment.

For every system, contact type and residue, the occupancy is defined as the
fraction of post-equilibration frames in which that contact is present.
Residue numbers are translated to the reference (equivalent) numbering via a
mapping built by :mod:`plfingerprint.mapping`, enabling cross-system
comparison. Contact residue numbers in Desmond exports are validated against
the UniProt sequence (native numbering, not RMSF order indices).
"""
import numpy as np
import pandas as pd

AA3_TO_1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
            "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
            "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
            "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
            "MSE": "M"}


def equivalent_resnum(iso, chain, resnum, resmap, iso_seq_keys,
                      reference_iso="alpha"):
    """Translate a residue number to the reference numbering scheme.

    Args:
        iso: isoform key of the system.
        chain: chain identifier (only chain "A" is mapped).
        resnum: native residue number.
        resmap: dict from :func:`mapping.map_uniprot_sequences`.
        iso_seq_keys: dict iso -> sequence key in ``resmap``.
        reference_iso: isoform used as the numbering reference.

    Returns:
        Equivalent residue number, or None if unmapped.
    """
    if chain != "A":
        return None
    key = iso_seq_keys.get(iso)
    if key is None or iso == reference_iso:
        return resnum
    return resmap.get(key, {}).get(resnum)


def contact_occupancy_table(data, systems, resmap, iso_seq_keys,
                            uniprot_seqs, iso_uniprot_keys,
                            ligands=None, reference_iso="alpha",
                            chain="A"):
    """Build the long-form contact occupancy table.

    One row per (system, contact type, residue). Occupancy is the fraction
    of all frames with the contact present. Residue identities are validated
    against the UniProt sequence; mismatches are counted and reported.

    Returns:
        (DataFrame, n_mismatches)
    """
    recs = []
    n_mismatch = 0
    for iso, pdbs in systems.items():
        useq = uniprot_seqs[iso_uniprot_keys[iso]]
        for pdb in pdbs:
            s = data[f"{iso}_{pdb}"]
            lig = (ligands or {}).get(pdb, "")
            for ctype, occ_df in s["contacts"].items():
                if ctype == "Metal" or len(occ_df) == 0:
                    continue
                grp = (occ_df.groupby(["chain", "idx", "resname"])["frame"]
                       .nunique() / s["n_frames"])
                for (ch, resnum, rn), occ in grp.items():
                    if ch != chain:
                        continue
                    if 1 <= resnum <= len(useq):
                        a1 = AA3_TO_1.get(rn)
                        if a1 is not None and a1 != useq[resnum - 1]:
                            n_mismatch += 1
                    recs.append(dict(
                        iso=iso, pdb=pdb, ligand=lig, type=ctype,
                        chain=ch, resname=rn, resnum=resnum,
                        ref_eq=equivalent_resnum(
                            iso, ch, resnum, resmap, iso_seq_keys,
                            reference_iso),
                        occupancy=occ))
    contacts = pd.DataFrame(recs)
    print(f"contacts: {len(contacts)} rows, "
          f"resname mismatches: {n_mismatch}", flush=True)
    return contacts, n_mismatch


def ligand_property_stats(data, systems, ligands=None,
                          equilibration_frames=200):
    """Mean/std of ligand properties (rGyr, SASA, PSA, MolSA, ...) over the
    equilibrated segment, plus ligand RMSF summary."""
    recs = []
    for iso, pdbs in systems.items():
        for pdb in pdbs:
            s = data[f"{iso}_{pdb}"]
            lp = s["l_props"]
            tail = lp[lp["frame"] >= equilibration_frames]
            d = dict(iso=iso, pdb=pdb,
                     ligand=(ligands or {}).get(pdb, ""))
            for c in ["RMSD", "rGyr", "intraHB", "MolSA", "SASA", "PSA"]:
                d[c + "_mean"] = tail[c].mean()
                d[c + "_std"] = tail[c].std()
            lr = s["l_rmsf"]
            d["L_RMSF_wrtLig_mean"] = lr["wrt_Ligand"].mean()
            d["L_RMSF_wrtLig_max"] = lr["wrt_Ligand"].max()
            d["L_RMSF_wrtProt_mean"] = lr["wrt_Protein"].mean()
            recs.append(d)
    return pd.DataFrame(recs)


def circular_std(deg):
    """Circular standard deviation of angles given in degrees."""
    rad = np.deg2rad(np.asarray(deg, dtype=float))
    R = np.hypot(np.cos(rad).mean(), np.sin(rad).mean())
    return np.rad2deg(np.sqrt(-2.0 * np.log(max(R, 1e-12))))


def torsion_flexibility_stats(data, systems, ligands=None,
                              equilibration_frames=200):
    """Per-dihedral circular std over the equilibrated segment."""
    recs = []
    for iso, pdbs in systems.items():
        for pdb in pdbs:
            s = data[f"{iso}_{pdb}"]
            t = s["torsions"]
            tail = t.iloc[equilibration_frames:]
            d = dict(iso=iso, pdb=pdb,
                     ligand=(ligands or {}).get(pdb, ""),
                     n_dihed=t.shape[1])
            for c in t.columns:
                d[c + "_cstd"] = circular_std(tail[c].values)
            d["mean_cstd"] = float(np.mean(
                [d[c + "_cstd"] for c in t.columns]))
            recs.append(d)
    return pd.DataFrame(recs)


def kinase_rmsf_table(data, systems, resmap, iso_seq_keys,
                      reference_iso="alpha", chain="A",
                      ref_range=(700, 1068)):
    """Per-residue C-alpha RMSF mapped to the reference numbering.

    Only residues whose equivalent number falls inside ``ref_range`` are
    kept.
    """
    recs = []
    lo, hi = ref_range
    for iso, pdbs in systems.items():
        for pdb in pdbs:
            s = data[f"{iso}_{pdb}"]
            ca = s["p_rmsf"][s["p_rmsf"]["chain"] == chain]
            for r in ca.itertuples():
                eq = equivalent_resnum(iso, chain, r.resnum, resmap,
                                       iso_seq_keys, reference_iso)
                if eq is not None and lo <= eq <= hi:
                    recs.append(dict(iso=iso, pdb=pdb, ref_eq=eq,
                                     resname=r.resname, resnum=r.resnum,
                                     CA=r.CA))
    return pd.DataFrame(recs)
