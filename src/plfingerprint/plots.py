"""Publication-style figures from interaction fingerprint analysis.

Functions
---------
Per-isoform figures (one PNG per isoform):
    rmsd_figure, rmsf_figure, occupancy_heatmap, ligand_property_bars,
    torsion_dials, fingerprint_radar
Cross-isoform comparison figures:
    compare_rmsf, compare_occupancy_heatmap, compare_ligprop_radar,
    compare_fingerprint_radar

All functions take explicit inputs (no globals) and write PNG files to a
given output directory. Residue numbers are expected in the reference
(equivalent) numbering scheme produced by :mod:`plfingerprint.mapping`.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.major.width": 0.8})

PDB_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
TYPE_LETTER = {"HBond": "H", "Hydrophobic": "B", "WaterBridge": "W",
               "Pi-Pi": "P", "Pi-Cation": "C", "Ionic": "I"}

FINGERPRINT_FEATURES = [
    ("Hinge Val851 H-bond", 851, "HBond"),
    ("Ser854 H-bond", 854, "HBond"),
    ("Lys802 contact", 802, "any"),
    ("Asp810 contact", 810, "any"),
    ("Asp933 water bridge", 933, "WaterBridge"),
    ("Trp780 pi-pi", 780, "Pi-Pi"),
    ("Tyr836 contact", 836, "any"),
    ("Ile800 hydrophobic", 800, "Hydrophobic"),
    ("Ile848 hydrophobic", 848, "Hydrophobic"),
    ("Ile932 hydrophobic", 932, "Hydrophobic"),
]

CONSERVED_POSITIONS = [780, 800, 802, 810, 836, 848, 849, 851,
                       854, 855, 856, 859, 919, 922, 932, 933]
CONSERVED_AA1 = {780: "W", 800: "I", 802: "K", 810: "D", 836: "Y",
                 848: "I", 849: "E", 851: "V", 854: "S", 855: "T",
                 856: "T", 859: "Q", 919: "S", 922: "M", 932: "I",
                 933: "D"}


def _radar_labels(ax, angles, labels, fontsize=8, r_lab=1.28):
    """Place radar axis labels outside the data polygon."""
    ax.set_xticks([])
    for ang, lab in zip(angles, labels):
        deg = float(np.degrees(ang)) % 360.0
        if deg < 15 or deg > 345:
            ha = "left"
        elif 165 < deg < 195:
            ha = "right"
        else:
            ha = "center"
        ax.text(ang, r_lab, lab, ha=ha, va="center", fontsize=fontsize)


def _iso_label(iso, ligands):
    return iso


def rmsd_figure(data, iso, pdbs, ligands, outdir, title=None):
    """Protein C-alpha and ligand RMSD time courses for one isoform."""
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.6), sharex=True)
    for pdb, c in zip(pdbs, PDB_COLORS):
        r = data[f"{iso}_{pdb}"]["rmsd"]
        lab = f"{pdb} ({ligands.get(pdb, '')})"
        axes[0].plot(r["time_ns"], r["Prot_CA"], color=c, lw=0.9, label=lab)
        axes[1].plot(r["time_ns"], r["Lig_wrt_Protein"], color=c, lw=0.9,
                     label=lab)
    for ax in axes:
        ax.set_xlim(0, r["time_ns"].max())
        ax.grid(alpha=0.2, lw=0.5)
    axes[0].set_ylabel("Protein C-alpha RMSD (A)")
    axes[1].set_ylabel("Ligand RMSD (A)")
    axes[1].set_xlabel("Time (ns)")
    axes[0].set_title(title or f"{iso} - RMSD time courses", fontsize=11)
    for ax, tag in zip(axes, "AB"):
        ax.text(0.01, 0.93, tag, transform=ax.transAxes, fontsize=12,
                fontweight="bold")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, fontsize=8, ncol=len(pdbs), framealpha=0.9,
               loc="upper center", bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(top=0.87, hspace=0.35)
    fig.savefig(os.path.join(outdir, f"rmsd_{iso}.png"), dpi=300)
    plt.close(fig)


def rmsf_figure(rmsf_df, iso, pdbs, ligands, outdir,
                ref_range=(720, 1060), title=None):
    """Kinase-domain C-alpha RMSF profiles (reference numbering)."""
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for pdb, c in zip(pdbs, PDB_COLORS):
        g = (rmsf_df[(rmsf_df["iso"] == iso) & (rmsf_df["pdb"] == pdb)]
             .sort_values("ref_eq").copy())
        g["grp"] = (g["ref_eq"].diff() != 1).cumsum()
        first = True
        for _, seg in g.groupby("grp"):
            ax.plot(seg["ref_eq"], seg["CA"], color=c, lw=0.9,
                    label=f"{pdb} ({ligands.get(pdb, '')})"
                    if first else None)
            first = False
    ax.set_xlim(*ref_range)
    for name, (a, b) in {"P-loop": (770, 778), "Hinge": (848, 856),
                         "DFG": (930, 936)}.items():
        ax.axvspan(a, b, color="gray", alpha=0.12)
        ax.text((a + b) / 2, 0.35, name, ha="center", va="bottom",
                fontsize=7, style="italic")
    ax.set_xlabel("Residue number (reference numbering)")
    ax.set_ylabel("C-alpha RMSF (A)")
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    ax.set_title(title or f"{iso} - C-alpha RMSF, kinase domain",
                 fontsize=11)
    ax.grid(alpha=0.2, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"rmsf_{iso}.png"), dpi=300)
    plt.close(fig)


def occupancy_heatmap(contacts_df, iso, pdbs, outdir, min_occupancy=0.30,
                      title=None):
    """Per-isoform contact occupancy heatmap (residues x systems)."""
    c = contacts_df[contacts_df["iso"] == iso].copy()
    key = c[c["occupancy"] >= min_occupancy].sort_values("ref_eq")
    residues = (key[["resname", "resnum", "ref_eq"]]
                .drop_duplicates().sort_values("ref_eq"))
    nres, nsys = len(residues), len(pdbs)
    mat = np.zeros((nres, nsys))
    letters = np.full((nres, nsys), "", dtype=object)
    res_list = list(residues.itertuples())
    for j, pdb in enumerate(pdbs):
        g = c[c["pdb"] == pdb]
        for i, r in enumerate(res_list):
            gg = g[g["ref_eq"] == r.ref_eq]
            if len(gg):
                best = gg.loc[gg["occupancy"].idxmax()]
                mat[i, j] = best["occupancy"]
                letters[i, j] = TYPE_LETTER.get(best["type"], "?")
    fig, ax = plt.subplots(figsize=(2.2 * nsys + 2, 0.42 * nres + 1.5))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(nsys))
    ax.set_xticklabels(pdbs, fontsize=8)
    ax.set_yticks(range(nres))
    ax.set_yticklabels([f"{r.resname}{r.ref_eq}" for r in res_list],
                       fontsize=7)
    for i in range(nres):
        for j in range(nsys):
            if mat[i, j] > 0.05:
                ax.text(j, i, f"{letters[i, j]}{mat[i, j]:.2f}",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if mat[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=ax, shrink=0.85).set_label(
        "Max contact proportion", fontsize=9)
    ax.set_title(title or f"{iso} - contact occupancy "
                 "(H=H-bond, B=hydrophobic, W=water bridge, P=pi-pi)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"occupancy_{iso}.png"), dpi=300)
    plt.close(fig)


def ligand_property_bars(ligstats_df, iso, pdbs, ligands, outdir,
                         title=None):
    """Grouped bars of ligand properties (mean +/- SD, equilibrated)."""
    props = [("rGyr", "rGyr (A)"), ("SASA", "SASA (A^2)"),
             ("PSA", "PSA (A^2)"), ("MolSA", "MolSA (A^2)")]
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5))
    for ax, (key, name) in zip(axes.flat, props):
        g = ligstats_df[ligstats_df["iso"] == iso].set_index("pdb").loc[pdbs]
        x = np.arange(len(pdbs))
        ax.bar(x, g[key + "_mean"], yerr=g[key + "_std"],
               color=PDB_COLORS[:len(pdbs)], capsize=3, edgecolor="black",
               lw=0.6, error_kw={"lw": 0.8})
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p}\n{ligands.get(p, '')}" for p in pdbs],
                           fontsize=7.5)
        ax.set_title(name, fontsize=10)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(title or f"{iso} - ligand properties "
                 "(mean +/- SD, equilibrated segment)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"ligprop_{iso}.png"), dpi=300)
    plt.close(fig)


def torsion_dials(data, iso, pdbs, ligands, outdir,
                  equilibration_frames=200, title=None):
    """Polar histograms of ligand torsion-angle distributions."""
    dials = {pdb: data[f"{iso}_{pdb}"]["torsions"].shape[1] for pdb in pdbs}
    maxd = max(dials.values())
    n = len(pdbs)
    fig = plt.figure(figsize=(2.2 * maxd + 1.5, 2.6 * n))
    gs = fig.add_gridspec(n, maxd + 1, width_ratios=[0.6] + [1] * maxd)
    for k, pdb in enumerate(pdbs):
        t = data[f"{iso}_{pdb}"]["torsions"]
        ax0 = fig.add_subplot(gs[k, 0])
        ax0.axis("off")
        ax0.text(0.5, 0.5,
                 f"{pdb}\n{ligands.get(pdb, '')}\n({dials[pdb]} dihedrals)",
                 ha="center", va="center", fontsize=9, fontweight="bold",
                 color=PDB_COLORS[k])
        for d in range(dials[pdb]):
            ax = fig.add_subplot(gs[k, d + 1], projection="polar")
            vals = np.deg2rad(t.iloc[equilibration_frames:, d].values)
            ax.hist(vals, bins=36, range=(-np.pi, np.pi),
                    color=PDB_COLORS[k], alpha=0.75)
            ax.set_title(f"dihed{d + 1}", fontsize=7, pad=4)
            ax.tick_params(labelsize=6)
            ax.set_yticklabels([])
    fig.suptitle(title or f"{iso} - ligand torsion angle distributions "
                 "(equilibrated segment)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"torsion_{iso}.png"), dpi=300)
    plt.close(fig)


def _fingerprint_values(contacts_df, iso, pdb,
                        features=FINGERPRINT_FEATURES):
    """Max proportion per fingerprint feature for one system."""
    c = contacts_df[(contacts_df["iso"] == iso) &
                    (contacts_df["pdb"] == pdb)]
    vals = []
    for _, eq, typ in features:
        g = c[c["ref_eq"] == eq]
        if typ == "any":
            v = g["occupancy"].max() if len(g) else 0
        else:
            gg = g[g["type"] == typ]
            v = gg["occupancy"].max() if len(gg) else 0
        vals.append(min(float(v) if pd.notna(v) else 0, 1.0))
    return vals


def fingerprint_radar(contacts_df, iso, pdbs, ligands, outdir,
                      features=FINGERPRINT_FEATURES, title=None):
    """Interaction fingerprint radar chart for the systems of one isoform."""
    labels = [f[0] for f in features]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    for pdb, col in zip(pdbs, PDB_COLORS):
        vals = _fingerprint_values(contacts_df, iso, pdb, features)
        vals = vals + vals[:1]
        ax.plot(angles, vals, color=col, lw=1.8,
                label=f"{pdb} ({ligands.get(pdb, '')})")
        ax.fill(angles, vals, color=col, alpha=0.08)
    _radar_labels(ax, angles[:-1], labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              fontsize=8, ncol=2)
    ax.set_title(title or f"{iso} - protein-ligand interaction fingerprint\n"
                 "(max proportion per feature)", fontsize=11, pad=55)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"radar_{iso}.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def compare_rmsf(rmsf_df, isos, iso_names, iso_colors, outdir, title=None):
    """Mean kinase-domain C-alpha RMSF per isoform on one axis."""
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for iso in isos:
        g = rmsf_df[rmsf_df["iso"] == iso]
        m = g.groupby("ref_eq")["CA"].mean()
        ax.plot(m.index.values, m.values, color=iso_colors[iso], lw=1.2,
                label=iso_names.get(iso, iso))
    for name, (a, b) in {"P-loop": (770, 778), "Hinge": (848, 856),
                         "DFG": (930, 936)}.items():
        ax.axvspan(a, b, color="gray", alpha=0.12)
        ax.text((a + b) / 2, 3.35, name, ha="center", fontsize=7,
                style="italic")
    ax.set_xlim(720, 1060)
    ax.set_ylim(0, 3.6)
    ax.set_xlabel("Residue number (reference numbering)")
    ax.set_ylabel("Mean C-alpha RMSF (A)")
    ax.legend(fontsize=9)
    ax.set_title(title or "Cross-isoform comparison - mean C-alpha RMSF "
                 "of the kinase domain", fontsize=11, pad=10)
    ax.grid(alpha=0.2, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "compare_rmsf.png"), dpi=300)
    plt.close(fig)


def compare_occupancy_heatmap(contacts_df, systems, isos, iso_names,
                              iso_colors, outdir,
                              conserved=CONSERVED_POSITIONS, title=None):
    """Contact fingerprint heatmap: conserved positions x all systems."""
    pairs = [(iso, pdb) for iso in isos for pdb in systems[iso]]
    nres, nsys = len(conserved), len(pairs)
    mat = np.zeros((nres, nsys))
    letters = np.full((nres, nsys), "", dtype=object)
    for j, (iso, pdb) in enumerate(pairs):
        g = contacts_df[(contacts_df["iso"] == iso) &
                        (contacts_df["pdb"] == pdb)]
        for i, eq in enumerate(conserved):
            gg = g[g["ref_eq"] == eq]
            if len(gg):
                best = gg.loc[gg["occupancy"].idxmax()]
                mat[i, j] = best["occupancy"]
                letters[i, j] = TYPE_LETTER.get(best["type"], "?")
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(nsys))
    ax.set_xticklabels([p for _, p in pairs], fontsize=7.5)
    pos = 0
    for iso in isos:
        n = len(systems[iso])
        ax.axvspan(pos - 0.5, pos + n - 0.5,
                   facecolor=iso_colors[iso], alpha=0.12)
        pos += n
    fig.canvas.draw()
    pos = 0
    for iso in isos:
        n = len(systems[iso])
        ax.text(pos + n / 2 - 0.5, -0.24, iso_names.get(iso, iso),
                ha="center", fontsize=10, fontweight="bold",
                color=iso_colors[iso],
                transform=ax.get_xaxis_transform(), clip_on=False)
        pos += n
    fig.subplots_adjust(bottom=0.22, top=0.90, left=0.07, right=0.92)
    ax.set_yticks(range(nres))
    ax.set_yticklabels([f"{CONSERVED_AA1.get(e, '?')}{e}" for e in conserved],
                       fontsize=8)
    for i in range(nres):
        for j in range(nsys):
            if mat[i, j] > 0.05:
                ax.text(j, i, f"{letters[i, j]}{mat[i, j]:.2f}",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if mat[i, j] > 0.55 else "black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Max contact proportion", fontsize=9)
    ax.set_title(title or "Cross-isoform contact fingerprint at conserved "
                 "positions (H=H-bond, B=hydrophobic, W=water bridge, "
                 "P=pi-pi)", fontsize=10, pad=18)
    fig.savefig(os.path.join(outdir, "compare_occupancy.png"), dpi=300)
    plt.close(fig)


def compare_ligprop_radar(ligstats_df, isos, iso_names, iso_colors, outdir,
                          title=None):
    """Normalized ligand-property radar, averaged per isoform."""
    props = ["rGyr_mean", "SASA_mean", "PSA_mean", "MolSA_mean"]
    labels = ["rGyr", "SASA", "PSA", "MolSA"]
    vals = ligstats_df[["iso"] + props].copy()
    for p in props:
        mn, mx = vals[p].min(), vals[p].max()
        vals[p] = (vals[p] - mn) / (mx - mn) if mx > mn else 0.5
    iso_mean = vals.groupby("iso")[props].mean()
    N = len(props)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    for iso in isos:
        if iso not in iso_mean.index:
            continue
        v = iso_mean.loc[iso].tolist() + iso_mean.loc[iso].tolist()[:1]
        ax.plot(angles, v, color=iso_colors[iso], lw=2,
                label=iso_names.get(iso, iso))
        ax.fill(angles, v, color=iso_colors[iso], alpha=0.1)
    _radar_labels(ax, angles[:-1], labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1.32, 0.78), fontsize=9)
    ax.set_title(title or "Ligand properties per isoform\n"
                 "(normalized; mean of systems)", fontsize=11, pad=55)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "compare_ligprop_radar.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def compare_fingerprint_radar(contacts_df, systems, isos, iso_names,
                              iso_colors, outdir,
                              features=FINGERPRINT_FEATURES, title=None):
    """Mean interaction fingerprint radar per isoform."""
    labels = [f[0] for f in features]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    for iso in isos:
        fps = np.array([_fingerprint_values(contacts_df, iso, pdb,
                                            features)
                        for pdb in systems[iso]])
        mean = fps.mean(axis=0)
        vals = mean.tolist() + mean.tolist()[:1]
        ax.plot(angles, vals, color=iso_colors[iso], lw=2,
                label=iso_names.get(iso, iso))
        ax.fill(angles, vals, color=iso_colors[iso], alpha=0.1)
    _radar_labels(ax, angles[:-1], labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
    ax.legend(loc="center left", bbox_to_anchor=(1.32, 0.78), fontsize=9)
    ax.set_title(title or "Mean interaction fingerprint per isoform\n"
                 "(max proportion per feature, averaged over systems)",
                 fontsize=11, pad=55)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "compare_fingerprint_radar.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
