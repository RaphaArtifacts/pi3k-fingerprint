"""Quickstart example: run the full fingerprint pipeline on 2 demo systems.

4JPS (PI3Kalpha + alpelisib): typical hinge-anchored binding mode.
5T8F (PI3Kdelta + taselisib): atypical binding mode (hinge H-bond absent).

Usage (from the repository root):
    pip install -r requirements.txt
    python examples/quickstart.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results")

cmd = [sys.executable, "-m", "plfingerprint", "run",
       "--config", os.path.join(HERE, "systems_example.yaml"),
       "--datadir", os.path.join(HERE, "data"),
       "--outdir", OUT,
       "--uniprot-json", os.path.join(HERE, "data", "uniprot.json")]
env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
print("running:", " ".join(cmd), flush=True)
r = subprocess.run(cmd, cwd=ROOT, env=env)
sys.exit(r.returncode)
