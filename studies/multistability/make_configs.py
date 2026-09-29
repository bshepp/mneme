"""Generate BETSE configs for the two multistability candidates.

Both follow the same design as studies/convergence: one tissue, fixed
parameters, several starting conditions, one exact repeat.

Method GRN (gene network coupled to voltage)
    Base: patterns_2018.yaml (BETSE doc/yaml/paper/2018_PBMB/Patterns).
    A cytosolic "Anion" inhibits a K+ leak channel, and gap junctions are
    voltage sensitive. Starting condition varied: the Anion's initial
    spatial distribution. Parameters, including the ion profile, are the
    paper's, so one seeded world is shared by all runs.

Method VGC (voltage-gated channels)
    Base: attractors_2016_1.yaml (as in studies/convergence) plus an
    inward-rectifier K+ channel (Kir2.1) and a Na+ leak, both active from
    the start. Starting condition varied: internal Na+/K+, as before. Three
    Na+ leak strengths are screened, each from a depolarised and a
    polarised start.

usage: make_configs.py <grn|vgc> <timing|full>
Run from a directory holding paper_grn.yaml / paper_vgc.yaml and their
geo/ and extra_configs/ directories.
"""
import sys
import warnings
from pathlib import Path

from ruamel.yaml import YAML

warnings.simplefilter("ignore")
method, mode = sys.argv[1], sys.argv[2]
yaml = YAML()
yaml.preserve_quotes = True
HERE = Path(".")

names = []


def export_only(c):
    """Turn off plots and animations; keep the per-cell Vmem CSV export."""
    r = c["results options"]
    for key in ("plot networks", "plot networks single cell", "plot cell cluster",
                "plot cell connectivity diagram", "plot cluster mask"):
        if key in r:
            r[key] = False
    r["while solving"]["animations"]["show"] = False
    r["while solving"]["animations"]["save"] = False
    r["after solving"]["plots"]["show"] = False
    r["after solving"]["plots"]["save"] = False
    r["after solving"]["animations"]["show"] = False
    r["after solving"]["animations"]["save"] = False
    r["save"]["data"]["all"]["enabled"] = False
    r["save"]["data"]["vmem"]["enabled"] = True


def set_paths(c, name, worldfile):
    c["init file saving"]["worldfile"] = worldfile
    c["init file saving"]["file"] = f"init_{name}.betse.gz"
    c["sim file saving"]["file"] = f"sim_{name}.betse.gz"
    c["results file saving"]["init directory"] = f"RESULTS/{name}/init"
    c["results file saving"]["sim directory"] = f"RESULTS/{name}/sim"


def set_times(c, init_total, sim_total, sample):
    c["init time settings"]["total time"] = init_total
    c["init time settings"]["sampling rate"] = sample
    c["sim time settings"]["total time"] = sim_total
    c["sim time settings"]["sampling rate"] = sample


if method == "grn":
    # Initial spatial distribution of the Anion. 'None' is uniform.
    STARTS = [
        ("bitmap", "gradient_bitmap"),   # the paper's starting condition
        ("gradx", "gradient_x"),
        ("grady", "gradient_y"),
        # gradient_r fails in BETSE 1.5.0 (array shape mismatch); a reversed
        # x gradient is used instead.
        ("gradx_rev", "gradient_x"),
        ("uniform", "None"),
    ]
    if mode == "timing":
        starts, init_total, sim_total, sample = STARTS[:1], 20.0, 20.0, 5.0
    else:
        starts = STARTS + [("bitmap_rep", "gradient_bitmap")]
        init_total, sim_total, sample = 500.0, 6000.0, 30.0

    for ic_label, asym in starts:
        name = f"{mode}_grn_ic_{ic_label}"
        c = yaml.load((HERE / "paper_grn.yaml").read_text(encoding="utf-8"))
        set_paths(c, name, f"world_{mode}_grn.betse.gz")   # shared world
        # Half the paper's world: about 240 cells instead of 970, four
        # times faster. The geometry bitmaps are scaled to the world.
        c["world options"]["world size"] = 500.0e-6
        set_times(c, init_total, sim_total, sample)
        export_only(c)

        # Per-run copy of the network config with the chosen start.
        net = yaml.load((HERE / "extra_configs" / "worm_3.yaml").read_text(encoding="utf-8"))
        anion = [m for m in net["biomolecules"] if m["name"] == "Anion"][0]
        anion["initial asymmetry"] = asym
        anion["plotting"]["plot 2D"] = False
        anion["plotting"]["animate"] = False
        net_path = HERE / "extra_configs" / f"worm_3_{name}.yaml"
        with open(net_path, "w", encoding="utf-8") as f:
            yaml.dump(net, f)
        c["gene regulatory network settings"]["gene regulatory network config"] = (
            f"extra_configs/worm_3_{name}.yaml"
        )
        if ic_label == "gradx_rev":
            c["modulator function properties"]["gradient_x"]["slope"] = -1.0
        with open(HERE / f"{name}.yaml", "w", encoding="utf-8") as f:
            yaml.dump(c, f)
        names.append(name)

elif method == "vgc":
    # (label, cytosolic Na+, cytosolic K+): a depolarised and a polarised start.
    STARTS = [("paper", 145.0, 5.0), ("na010", 10.0, 140.0)]
    # Na+ leak strength [m^2/s]; the base membrane Na+ permeability is 7.5e-19.
    LEAKS = [("naL3e18", 3.0e-18), ("naL1e17", 1.0e-17), ("naL3e17", 3.0e-17)]
    KIR_DM = 5.0e-17   # Kir2.1 fully-open permeability; base K+ leak is 1.5e-17
    if mode == "timing":
        starts, leaks, init_total, sim_total, sample = STARTS[:1], LEAKS[1:2], 20.0, 20.0, 5.0
    else:
        starts, leaks = STARTS + [("paper_rep", 145.0, 5.0)], LEAKS
        init_total, sim_total, sample = 3600.0, 21600.0, 60.0

    for leak_label, leak_dm in leaks:
        for ic_label, na, k in starts:
            if ic_label == "paper_rep" and leak_label != "naL1e17":
                continue
            name = f"{mode}_vgc_{leak_label}_ic_{ic_label}"
            c = yaml.load((HERE / "paper_vgc.yaml").read_text(encoding="utf-8"))
            # Starting ions are tied to the seed, so each run is seeded
            # separately; lattice disorder 0 makes the seeds identical.
            set_paths(c, name, f"world_{name}.betse.gz")
            c["world options"]["lattice disorder"] = 0.0
            c["world options"]["world size"] = 80.0e-6
            set_times(c, init_total, sim_total, sample)
            export_only(c)
            ion = c["general options"]["customized ion profile"]
            ion["cytosolic Na+ concentration"] = na
            ion["cytosolic K+ concentration"] = k
            c["general network"]["channels"] = [
                {"name": "Kir", "channel class": "K", "channel type": "Kir2p1",
                 "max Dm": KIR_DM, "apply to": "all", "init active": True},
                {"name": "NaLeak", "channel class": "Na", "channel type": "NaLeak",
                 "max Dm": leak_dm, "apply to": "all", "init active": True},
            ]
            with open(HERE / f"{name}.yaml", "w", encoding="utf-8") as f:
                yaml.dump(c, f)
            names.append(name)
else:
    raise SystemExit("method must be grn or vgc")

(HERE / f"{mode}_{method}_runs.txt").write_text("\n".join(names) + "\n", encoding="utf-8", newline="\n")
print("\n".join(names))
