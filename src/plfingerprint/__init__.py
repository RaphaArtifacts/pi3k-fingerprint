"""plfingerprint: dynamic protein-ligand interaction fingerprinting from MD data.

Parses Desmond molecular dynamics analysis exports (.dat), maps residues to a
common reference numbering across protein isoforms/families, computes contact
occupancy statistics over the equilibrated trajectory segment, and generates
comparative fingerprint figures (heatmaps, radar plots, RMSF profiles).
"""

__version__ = "1.0.0"
