"""Generate BETSE configs for the basin-mapping study: the gene-network
tissue started from random Anion fields.

usage: make_configs.py <n_seeds>

Writes geo/random_L<len>_r<seed>.png bitmaps and one config per bitmap,
basin_L<len>_r<seed>.yaml. The Anion's initial concentration in each cell is
the paper's 1.0 mmol/L times the bitmap's red channel at the cell (0 to 1),
the same mechanism as the paper's own bitmap gradient start. Each bitmap is
Gaussian-filtered white noise with correlation length <len> um (1 px = 1 um
in the 500 um world), scaled to [0, 1].

Runs share the perturbation study's world file and need only the init phase:
    PHASES="init,plot init" SKIP_SEED=1 bash ../../convergence/run_one.sh <name> .
Run from a directory holding paper_grn.yaml, geo/ and extra_configs/.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grn_common import grn_config, yaml  # noqa: E402

HERE = Path(".")
WORLD = "world_perturb_grn.betse.gz"    # shared with studies/perturbation
INIT_S, SAMPLE_S = 6000.0, 30.0
LENGTHS_UM = [25, 100]                   # correlation lengths of the random field
IMAGE_PX = 500                           # the paper's bitmaps are 500 x 500


def random_bitmap(length_um, seed, path):
    rng = np.random.default_rng(seed)
    u = gaussian_filter(rng.standard_normal((IMAGE_PX, IMAGE_PX)), sigma=length_um, mode="wrap")
    u = (u - u.min()) / (u.max() - u.min())
    rgb = np.zeros((IMAGE_PX, IMAGE_PX, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.round(255 * u).astype(np.uint8)   # red = positive, blue = negative
    Image.fromarray(rgb, "RGB").save(path)
    return float(u.mean()), float(u.std())


def main(n_seeds):
    names = []
    for length in LENGTHS_UM:
        for seed in range(1, n_seeds + 1):
            name = f"basin_L{length}_r{seed}"
            bitmap = f"geo/random_L{length}_r{seed}.png"
            mean, std = random_bitmap(length, seed, HERE / bitmap)
            c = grn_config(name, asym="gradient_bitmap", bitmap=bitmap, world=WORLD,
                           init_total=INIT_S, sim_total=60.0, sample=SAMPLE_S, here=HERE)
            with open(HERE / f"{name}.yaml", "w", encoding="utf-8") as f:
                yaml.dump(c, f)
            names.append(name)
            print(f"{name}: bitmap mean {mean:.3f}, sd {std:.3f}")
    (HERE / "basin_runs.txt").write_text("\n".join(names) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
