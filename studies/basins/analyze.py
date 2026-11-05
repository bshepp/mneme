"""Analyse the basin-mapping runs: which pattern does each random start reach?

usage: python studies/basins/analyze.py <run_root> [--out results.json]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grn_endstates import analyze  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    analyze(a.run_root, prefix="basin_", group=lambda n: "L" + n.split("_L")[1].split("_")[0],
            out=a.out)
