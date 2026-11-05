"""Shared config helpers for the gene-network (GRN) BETSE studies.

The GRN studies (multistability's GRN method, perturbation, basins, sweep)
all start from the 2018 patterns configuration, halve the world, disable the
paper's tissue cut, and vary only the Anion's initial distribution or one
parameter. This module holds that common setup.

Scripts import it with::

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from grn_common import grn_config, yaml

and run from a directory holding paper_grn.yaml (patterns_2018.yaml) with
its geo/ and extra_configs/ directories; see studies/betse_configs/README.md.
"""
from __future__ import annotations

import warnings
from pathlib import Path

from ruamel.yaml import YAML

warnings.simplefilter("ignore")
yaml = YAML()
yaml.preserve_quotes = True

WORLD_SIZE_M = 500.0e-6    # half the paper's world: about 250 cells


def export_only(c):
    """Turn off plots and animations; keep the per-cell Vmem CSV export."""
    r = c["results options"]
    for key in ("plot networks", "plot networks single cell", "plot cell cluster",
                "plot cell connectivity diagram", "plot cluster mask"):
        if key in r:
            r[key] = False
    for phase in ("while solving", "after solving"):
        for kind in ("plots", "animations"):
            if kind in r[phase]:
                r[phase][kind]["show"] = False
                r[phase][kind]["save"] = False
    r["save"]["data"]["all"]["enabled"] = False
    r["save"]["data"]["vmem"]["enabled"] = True


def set_paths(c, name, worldfile, init_name=None):
    c["init file saving"]["worldfile"] = worldfile
    c["init file saving"]["file"] = f"init_{init_name or name}.betse.gz"
    c["sim file saving"]["file"] = f"sim_{name}.betse.gz"
    c["results file saving"]["init directory"] = f"RESULTS/{init_name or name}/init"
    c["results file saving"]["sim directory"] = f"RESULTS/{name}/sim"


def set_times(c, init_total, sim_total, sample):
    c["init time settings"]["total time"] = init_total
    c["init time settings"]["sampling rate"] = sample
    c["sim time settings"]["total time"] = sim_total
    c["sim time settings"]["sampling rate"] = sample


def grn_config(name, *, asym, world, init_total, sim_total, sample,
               slope=None, bitmap=None, network=None, gj_surface=None, here=Path(".")):
    """One GRN run config, written as <name>.yaml by the caller.

    asym        the Anion's 'initial asymmetry': gradient_bitmap, gradient_x,
                gradient_y or None (uniform)
    slope       for gradient_x: slope on top of a uniform Anion (z-offset 1)
    bitmap      for gradient_bitmap: image file relative to the run dir
    network     dict of overrides for the Anion entry of the network file,
                e.g. {"Dgj": 1e-14}
    gj_surface  the main config's 'gap junction surface area' (paper: 1e-8)
    """
    c = yaml.load((here / "paper_grn.yaml").read_text(encoding="utf-8"))
    set_paths(c, name, world)
    c["world options"]["world size"] = WORLD_SIZE_M
    set_times(c, init_total, sim_total, sample)
    export_only(c)
    net = yaml.load((here / "extra_configs" / "worm_3.yaml").read_text(encoding="utf-8"))
    anion = [m for m in net["biomolecules"] if m["name"] == "Anion"][0]
    anion["initial asymmetry"] = asym
    anion["plotting"]["plot 2D"] = False
    anion["plotting"]["animate"] = False
    for key, value in (network or {}).items():
        anion[key] = value
    net_path = here / "extra_configs" / f"worm_3_{name}.yaml"
    with open(net_path, "w", encoding="utf-8") as f:
        yaml.dump(net, f)
    c["gene regulatory network settings"]["gene regulatory network config"] = (
        f"extra_configs/worm_3_{name}.yaml")
    c["cutting event"]["event happens"] = False   # the paper's cut; runs are undisturbed
    if slope is not None:
        c["modulator function properties"]["gradient_x"]["slope"] = slope
        c["modulator function properties"]["gradient_x"]["z-offset"] = 1.0
    if bitmap is not None:
        c["modulator function properties"]["gradient_bitmap"]["file"] = bitmap
    if gj_surface is not None:
        c["variable settings"]["gap junctions"]["gap junction surface area"] = gj_surface
    return c
