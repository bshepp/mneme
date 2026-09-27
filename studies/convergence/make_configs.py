"""Generate BETSE configs for the convergence / multistability experiment.

All runs share ONE world (same cells, same geometry) and one parameter set.
They differ only in the starting intracellular Na+ and K+ concentrations,
and, in the coupling sweep, in gap-junction surface area.

usage: make_configs.py <mode>      mode = timing | full
"""
import sys
import warnings
from pathlib import Path

from ruamel.yaml import YAML

warnings.simplefilter("ignore")
mode = sys.argv[1]
yaml = YAML()
yaml.preserve_quotes = True
HERE = Path(__file__).parent

# (label, cytosolic Na+, cytosolic K+) in mmol/L. Protein and Cl- are held
# at the paper's values, so every run is the same system.
INITIAL_CONDITIONS = [
    ("ic_paper", 145.0, 5.0),     # the published starting point: inside == outside
    ("ic_na120", 120.0, 30.0),
    ("ic_na100", 100.0, 50.0),
    ("ic_equal", 75.0, 75.0),
    ("ic_na050", 50.0, 100.0),
    ("ic_na010", 10.0, 140.0),    # near-physiological gradients from the start
]
# Gap-junction surface area [m^2]. 1e-15 is the paper's value (cells
# effectively uncoupled). 1e-9 is the strongest coupling tested that stayed
# numerically stable at this time step; BETSE's default of 5e-8 did not.
COUPLINGS = [("gj_off", 1.0e-15), ("gj_on", 1.0e-9)]

if mode == "timing":
    ics, couplings = INITIAL_CONDITIONS[:1], COUPLINGS
    init_total, sim_total, sample = 20.0, 20.0, 5.0
else:
    ics, couplings = INITIAL_CONDITIONS, COUPLINGS
    init_total, sim_total, sample = 3600.0, 21600.0, 60.0

names = []
for gj_label, gj_area in couplings:
    run_ics = list(ics)
    if mode == "full" and gj_label == "gj_off":
        # Exact repeat of one run, to measure replicate noise.
        run_ics.append(("ic_paper_rep", 145.0, 5.0))
    for ic_label, na, k in run_ics:
        name = f"{mode}_{gj_label}_{ic_label}"
        c = yaml.load((HERE / "paper.yaml").read_text(encoding="utf-8"))

        # BETSE ties the starting ion concentrations to the seeded world, so
        # each run is seeded separately. Lattice disorder is set to zero so
        # that every seed produces the same cells (checked after the runs).
        c["init file saving"]["worldfile"] = f"world_{name}.betse.gz"
        c["world options"]["lattice disorder"] = 0.0
        c["init file saving"]["file"] = f"init_{name}.betse.gz"
        c["sim file saving"]["file"] = f"sim_{name}.betse.gz"
        c["results file saving"]["init directory"] = f"RESULTS/{name}/init"
        c["results file saving"]["sim directory"] = f"RESULTS/{name}/sim"

        c["init time settings"]["total time"] = init_total
        c["init time settings"]["sampling rate"] = sample
        c["sim time settings"]["total time"] = sim_total
        c["sim time settings"]["sampling rate"] = sample

        c["world options"]["world size"] = 80.0e-6

        ion = c["general options"]["customized ion profile"]
        ion["cytosolic Na+ concentration"] = na
        ion["cytosolic K+ concentration"] = k

        c["variable settings"]["gap junctions"]["gap junction surface area"] = gj_area

        # Data export only: no plots or animations.
        r = c["results options"]
        r["plot networks"] = False
        r["plot networks single cell"] = False
        r["plot cell cluster"] = False
        r["plot cell connectivity diagram"] = False
        r["plot cluster mask"] = False
        r["while solving"]["animations"]["show"] = False
        r["while solving"]["animations"]["save"] = False
        r["after solving"]["plots"]["show"] = False
        r["after solving"]["plots"]["save"] = False
        r["after solving"]["animations"]["show"] = False
        r["after solving"]["animations"]["save"] = False
        r["save"]["data"]["all"]["enabled"] = False
        r["save"]["data"]["vmem"]["enabled"] = True

        with open(HERE / f"{name}.yaml", "w", encoding="utf-8") as f:
            yaml.dump(c, f)
        names.append(name)

(HERE / f"{mode}_runs.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
print("\n".join(names))
