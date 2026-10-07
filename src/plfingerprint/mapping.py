"""Cross-isoform equivalent residue mapping.

Builds a residue-level correspondence between protein isoforms (or family
members) so that interaction fingerprints from different systems can be
compared on a common numbering scheme. The reference is typically the most
studied isoform (here human PIK3CA).

Two mapping routes are provided:
1. :func:`map_uniprot_sequences` -- pairwise-align full-length UniProt
   sequences to the reference sequence (recommended; numbering matches the
   canonical UniProt residue numbers).
2. :func:`map_pdb_seqres` -- pairwise-align PDB SEQRES records to the
   reference PDB sequence (useful when UniProt mapping is unavailable).

Alignments use Biopython's PairwiseAligner with BLOSUM62.
"""
import json
import pickle

from Bio.Align import PairwiseAligner, substitution_matrices

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
       "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
       "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
       "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
       "MSE": "M", "UNK": "X"}


def make_aligner():
    """Return a PairwiseAligner configured for protein sequence alignment."""
    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -1
    aligner.target_end_gap_score = 0
    aligner.query_end_gap_score = 0
    return aligner


def alignment_to_map(aln, one_based=True):
    """Convert a pairwise alignment to a {query_pos: ref_pos} dict.

    Only aligned (non-gap) blocks are mapped. Positions are 1-based by
    default to match UniProt/PDB residue numbering.
    """
    rc, qc = aln.coordinates[0], aln.coordinates[1]
    offset = 1 if one_based else 0
    q2r = {}
    for i in range(len(rc) - 1):
        r0, r1 = int(rc[i]), int(rc[i + 1])
        q0, q1 = int(qc[i]), int(qc[i + 1])
        if r1 > r0 and q1 > q0:  # aligned block
            for k in range(q1 - q0):
                q2r[q0 + k + offset] = r0 + k + offset
    return q2r


def map_uniprot_sequences(uniprot_seqs, reference_key, query_keys,
                          aligner=None):
    """Map UniProt residue numbers of query isoforms to the reference.

    Args:
        uniprot_seqs: dict mapping sequence key -> full amino-acid string.
        reference_key: key of the reference sequence in ``uniprot_seqs``.
        query_keys: list of keys to map onto the reference.
        aligner: optional pre-configured PairwiseAligner.

    Returns:
        dict mapping query key -> {query_resnum: ref_resnum}.
    """
    aligner = aligner or make_aligner()
    ref = uniprot_seqs[reference_key]
    maps = {}
    for key in query_keys:
        aln = aligner.align(ref, uniprot_seqs[key])[0]
        maps[key] = alignment_to_map(aln, one_based=True)
        print(f"{key}: {len(maps[key])} residues mapped "
              f"of {len(uniprot_seqs[key])}", flush=True)
    return maps


def seqres_chain(pdbfile, chain="A"):
    """Extract the SEQRES amino-acid sequence of one chain from a PDB file."""
    seq = []
    with open(pdbfile) as fh:
        for line in fh:
            if line.startswith("SEQRES") and line[11] == chain:
                for p in line[19:].split():
                    seq.append(AA3.get(p, "X"))
    return "".join(seq)


def map_pdb_seqres(pdb_files, reference_pdb, chain="A", aligner=None):
    """Map PDB SEQRES positions of query structures to the reference PDB.

    Args:
        pdb_files: dict mapping a structure id -> PDB file path.
        reference_pdb: id of the reference structure in ``pdb_files``.
        chain: chain identifier to use (default ``"A"``).

    Returns:
        dict mapping structure id -> {query_seqpos: ref_seqpos} (0-based).
    """
    aligner = aligner or make_aligner()
    seqs = {pid: seqres_chain(p, chain) for pid, p in pdb_files.items()}
    ref = seqs[reference_pdb]
    maps = {}
    for pid, seq in seqs.items():
        if pid == reference_pdb:
            continue
        aln = aligner.align(ref, seq)[0]
        maps[pid] = alignment_to_map(aln, one_based=False)
        ident = sum(1 for a, b in zip(aln[0], aln[1])
                    if a == b and a != "-") / len(ref)
        print(f"{pid}: aligned, identity ~{ident:.2f}", flush=True)
    return maps


def save_maps(maps, path):
    """Pickle a mapping dict to ``path``."""
    with open(path, "wb") as fh:
        pickle.dump(maps, fh)


def load_maps(path):
    """Load a pickled mapping dict from ``path``."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


def save_sequences(seqs, path):
    """Save {key: sequence} as JSON."""
    with open(path, "w") as fh:
        json.dump(seqs, fh)


def load_sequences(path):
    """Load {key: sequence} from JSON."""
    with open(path) as fh:
        return json.load(fh)
