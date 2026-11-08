"""Analyse the coupling sweep: how many end states at each coupling value?

usage: python studies/sweep/analyze.py <run_root> [--out results.json]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grn_endstates import analyze

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    analyze(a.run_root, prefix="sweep_", group=lambda n: n.rsplit("_", 1)[0], out=a.out)
