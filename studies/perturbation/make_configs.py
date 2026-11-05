"""Generate BETSE configs for the perturbation study (protocol step 4).

The gene-network (GRN) configuration of studies/multistability has two
stable voltage patterns. This study settles the tissue into each, displaces
the settled state by a stated amount, and continues the run.

usage: make_configs.py settle            # two settling runs: pattern A and B
       make_configs.py perturb           # perturbed continuations of both
       make_configs.py continue <run> <seconds>   # carry a finished run on

Run from a directory holding paper_grn.yaml (patterns_2018.yaml) with its
geo/ and extra_configs/ directories; see studies/betse_configs/README.md.
``perturb`` needs the settling runs' init files (INITS/init_settle_*.betse.gz)
and writes a displaced copy of each for every perturbation, plus a sim-only
config that continues from it. Run those with PHASES="sim,plot sim".

Pattern A is reached from the paper's bitmap Anion gradient; pattern B from a
uniform Anion plus a 1% x gradient (studies/multistability/RESULTS.md).

The perturbations act on the state saved at the end of the settling init.
The state of this system is the per-cell Anion concentration (slow, the
pattern's memory) and the per-cell voltage (fast, its readout):

  ctrl        the saved state written back unchanged (round-trip control)
  erase_fF    in the depolarised cells, move the Anion F of the way to the
              mean Anion of the polarised cells (F = 25, 50, 100 %)
  toward_fF   move every cell's Anion F of the way to the other pattern's
              (F = 25, 50, 75 %)
  noise_sS_rR multiply every cell's Anion by exp(S * z), z ~ N(0, 1), seed R
              (S = 0.25, 0.5, 1.0; R = 1, 2)
  vm_only     shift the depolarised cells' voltage 20 mV toward the
              polarised value, Anion untouched (short run)

``continue`` takes the state saved at the end of a finished run's sim phase
(SIMS/sim_<run>.betse.gz) as the init file of a new run cont_<run>, and runs
a sim phase of the given length from it. Used to check whether a run that
had not settled, or settled somewhere new, stays there.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grn_common import grn_config, yaml

HERE = Path(".")

SETTLE_INIT_S = 6000.0     # settling, as the init phase (about 15 relaxation times)
SETTLE_SIM_S = 3000.0      # unperturbed continuation of the settled state
PERTURB_SIM_S = 3000.0     # continuation after a perturbation
VM_ONLY_SIM_S = 600.0
SAMPLE_S = 30.0
WORLD = "world_perturb_grn.betse.gz"

# (label, Anion initial asymmetry, gradient_x slope or None)
PATTERNS = {"A": ("gradient_bitmap", None), "B": ("gradient_x", 0.01)}

ERASE_FRACTIONS = [0.25, 0.5, 1.0]
TOWARD_FRACTIONS = [0.25, 0.5, 0.75]
NOISE_SCALES = [0.25, 0.5, 1.0]
NOISE_SEEDS = [1, 2]
VM_SHIFT_V = 0.020


def settle_config(label):
    """The settling run for one pattern: the multistability GRN setup."""
    asym, slope = PATTERNS[label]
    name = f"settle_{label}"
    c = grn_config(name, asym=asym, slope=slope, world=WORLD, init_total=SETTLE_INIT_S,
                   sim_total=SETTLE_SIM_S, sample=SAMPLE_S, here=HERE)
    return name, c


def main(mode):
    names = []
    if mode == "settle":
        for label in PATTERNS:
            name, c = settle_config(label)
            with open(HERE / f"{name}.yaml", "w", encoding="utf-8") as f:
                yaml.dump(c, f)
            names.append(name)

    elif mode == "perturb":
        from betse.science import filehandling as fh

        def load(label):
            return fh.loadSim(str(HERE / "INITS" / f"init_settle_{label}.betse.gz"))

        def anion(sim):
            return sim.grn.core.molecules["Anion"].c_cells

        # The other pattern's Anion field, for the "toward" perturbations.
        other_anion = {label: np.array(anion(load(label)[0])) for label in PATTERNS}
        record = {}
        for label in PATTERNS:
            other = [k for k in PATTERNS if k != label][0]
            base_sim, base_cells, _ = load(label)
            vm = base_sim.vm_ave * 1e3
            thr = 0.5 * (vm.min() + vm.max())
            depol = vm > thr
            if depol.sum() == 0 or (~depol).sum() == 0:
                raise SystemExit(f"pattern {label} is not bimodal: {vm.min():.1f} to {vm.max():.1f} mV")
            c0 = np.array(anion(base_sim))
            pol_anion = float(c0[~depol].mean())
            xy = base_cells.cell_centres * 1e6
            record[label] = dict(
                n_cells=int(len(vm)), n_depolarised=int(depol.sum()), threshold_mV=float(thr),
                vm_depolarised_mV=float(vm[depol].mean()), vm_polarised_mV=float(vm[~depol].mean()),
                anion_depolarised=float(c0[depol].mean()), anion_polarised=pol_anion,
                depolarised_cells=np.flatnonzero(depol).tolist(),
                x_um=xy[:, 0].round(3).tolist(), y_um=xy[:, 1].round(3).tolist(), runs={},
            )

            perts = [("ctrl", None, None)]
            perts += [(f"erase_f{int(f * 100)}", "erase", f) for f in ERASE_FRACTIONS]
            perts += [(f"toward_f{int(f * 100)}", "toward", f) for f in TOWARD_FRACTIONS]
            perts += [(f"noise_s{str(s).replace('.', 'p')}_r{r}", "noise", (s, r))
                      for s in NOISE_SCALES for r in NOISE_SEEDS]
            perts += [("vm_only", "vm", VM_SHIFT_V)]

            for plabel, kind, par in perts:
                sim, cells, p = load(label)          # a fresh copy each time
                c = anion(sim)
                vshift = np.zeros_like(sim.vm_ave)
                if kind == "erase":
                    c[depol] += par * (pol_anion - c[depol])
                elif kind == "toward":
                    c[:] = c + par * (other_anion[other] - c)
                elif kind == "noise":
                    s, r = par
                    z = np.random.default_rng(r).standard_normal(len(c))
                    c[:] = c * np.exp(s * z)
                elif kind == "vm":
                    vshift[depol] = -par
                    sim.vm_ave = sim.vm_ave + vshift
                    sim.vm = sim.vm_ave[cells.mem_to_cells]
                name = f"pert_{label}_{plabel}"
                fh.saveSim(str(HERE / "INITS" / f"init_{name}.betse.gz"), [sim, cells, p])

                _, cfg = settle_config(label)
                cfg["init file saving"]["file"] = f"init_{name}.betse.gz"
                cfg["sim file saving"]["file"] = f"sim_{name}.betse.gz"
                cfg["results file saving"]["init directory"] = f"RESULTS/{name}/init"
                cfg["results file saving"]["sim directory"] = f"RESULTS/{name}/sim"
                cfg["sim time settings"]["total time"] = (
                    VM_ONLY_SIM_S if kind == "vm" else PERTURB_SIM_S)
                with open(HERE / f"{name}.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f)
                names.append(name)
                rel = np.abs(c - c0) / c0
                record[label]["runs"][name] = dict(
                    kind=kind or "ctrl", parameter=par,
                    anion_max_relative_change=float(rel.max()),
                    anion_mean_relative_change=float(rel.mean()),
                    n_cells_changed=int((rel > 1e-12).sum() + (vshift != 0).sum()),
                    vm_shift_mV=float(-vshift.min() * 1e3) if kind == "vm" else 0.0,
                )
        (HERE / "perturbations.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    elif mode == "continue":
        import shutil
        run, seconds = sys.argv[2], float(sys.argv[3])
        label = run.split("_")[1] if run.startswith("pert_") else run.split("_")[1]
        name = f"cont_{run}"
        shutil.copyfile(HERE / "SIMS" / f"sim_{run}.betse.gz", HERE / "INITS" / f"init_{name}.betse.gz")
        _, cfg = settle_config(label)
        cfg["init file saving"]["file"] = f"init_{name}.betse.gz"
        cfg["sim file saving"]["file"] = f"sim_{name}.betse.gz"
        cfg["results file saving"]["init directory"] = f"RESULTS/{name}/init"
        cfg["results file saving"]["sim directory"] = f"RESULTS/{name}/sim"
        cfg["sim time settings"]["total time"] = seconds
        with open(HERE / f"{name}.yaml", "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
        names.append(name)
    else:
        raise SystemExit("mode must be settle, perturb or continue")

    (HERE / f"{mode}_runs.txt").write_text("\n".join(names) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(names))


if __name__ == "__main__":
    main(sys.argv[1])
