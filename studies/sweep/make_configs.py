"""Generate BETSE configs for the coupling sweep (protocol step 5) on the
gene-network tissue.

usage: make_configs.py <dgj|gj>

dgj  sweeps the Anion's gap-junction diffusion constant Dgj, the paper's own
     "master variable" for the pattern (paper value 5e-14, suggested range
     1e-15 to 1e-13 m^2/s). This sets how fast the pattern's memory variable
     moves between cells.
gj   sweeps the main config's gap junction surface area, which sets the ionic
     coupling between cells (paper value 1e-8, suggested range 1e-9 to 1e-7).

At each value the tissue is started from the two conditions that reached
patterns A and B at the paper's values: the paper's bitmap Anion gradient,
and a uniform Anion plus a 1% x gradient. Runs share the perturbation
study's world and need only the init phase:
    PHASES="init,plot init" SKIP_SEED=1 bash ../../convergence/run_one.sh <name> .
Run from a directory holding paper_grn.yaml, geo/ and extra_configs/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grn_common import grn_config, yaml

HERE = Path(".")
WORLD = "world_perturb_grn.betse.gz"    # shared with studies/perturbation
INIT_S, SAMPLE_S = 6000.0, 30.0
STARTS = {"A": ("gradient_bitmap", None), "B": ("gradient_x", 0.01)}
DGJ_VALUES = [5.0e-15, 2.0e-14, 1.5e-13]          # paper: 5e-14, already run
GJ_SURFACE_VALUES = [1.0e-9, 1.0e-7]              # paper: 1e-8, already run


def label(v):
    return f"{v:.0e}".replace("e-", "em").replace("+", "")


def main(which):
    names = []
    if which == "dgj":
        for v in DGJ_VALUES:
            for start, (asym, slope) in STARTS.items():
                name = f"sweep_dgj{label(v)}_{start}"
                c = grn_config(name, asym=asym, slope=slope, world=WORLD, network={"Dgj": v},
                               init_total=INIT_S, sim_total=60.0, sample=SAMPLE_S, here=HERE)
                with open(HERE / f"{name}.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(c, f)
                names.append(name)
    elif which == "gj":
        for v in GJ_SURFACE_VALUES:
            for start, (asym, slope) in STARTS.items():
                name = f"sweep_gj{label(v)}_{start}"
                c = grn_config(name, asym=asym, slope=slope, world=WORLD, gj_surface=v,
                               init_total=INIT_S, sim_total=60.0, sample=SAMPLE_S, here=HERE)
                with open(HERE / f"{name}.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(c, f)
                names.append(name)
    else:
        raise SystemExit("argument must be dgj or gj")
    (HERE / f"sweep_{which}_runs.txt").write_text("\n".join(names) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(names))


if __name__ == "__main__":
    main(sys.argv[1])
