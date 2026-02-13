#!/usr/bin/env python
"""Analyze BETSE bioelectric simulation output with the Mneme pipeline.

This script loads Vmem field data exported by BETSE, runs it through
Mneme's field analysis pipeline (topology, attractors, field reconstruction),
and saves the results.

Usage
-----
    python scripts/analyze_betse.py <vmem_dir> [--resolution 64] [--output results/betse]

Example
-------
    python scripts/analyze_betse.py \\
        "F:/consciousness-projects/betse/doc/yaml/paper/sample_sim/RESULTS/sim_1/Vmem2D_TextExport" \\
        --resolution 64 \\
        --output results/betse_sample
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mneme.data.betse_loader import (
    betse_to_field,
    load_betse_timeseries,
    load_betse_exported_data,
)
from mneme.types import Field
from mneme.core.topology import PersistentHomology, field_to_point_cloud
from mneme.core.attractors import (
    embed_trajectory,
    compute_lyapunov_spectrum,
    classify_attractor_by_lyapunov,
    kaplan_yorke_dimension,
    RecurrenceAnalysis,
)
from mneme.utils.io import save_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("analyze_betse")


def analyze_single_frame(field_2d: np.ndarray, label: str = "frame") -> dict:
    """Run topology analysis on a single 2-D Vmem field."""
    results = {"label": label}

    # --- Persistent homology (cubical complex) ---
    try:
        ph = PersistentHomology(max_dimension=2, persistence_threshold=0.01)
        diagrams = ph.compute_persistence(field_2d)
        features = ph.extract_features(diagrams) if diagrams else None

        n_features = sum(len(d.points) for d in diagrams) if diagrams else 0
        results["topology"] = {
            "n_diagrams": len(diagrams) if diagrams else 0,
            "total_features": n_features,
            "feature_vector_length": len(features) if features is not None else 0,
        }
        if diagrams:
            for d in diagrams:
                dim = d.dimension
                if len(d.points) > 0:
                    persistence = d.points[:, 1] - d.points[:, 0]
                    results["topology"][f"dim{dim}_count"] = len(d.points)
                    results["topology"][f"dim{dim}_max_persistence"] = float(
                        persistence.max()
                    )
                    results["topology"][f"dim{dim}_mean_persistence"] = float(
                        persistence.mean()
                    )
        logger.info(
            "  %s topology: %d total features across %d diagrams",
            label,
            n_features,
            len(diagrams) if diagrams else 0,
        )
    except Exception as exc:
        logger.warning("  Topology analysis failed for %s: %s", label, exc)
        results["topology"] = {"error": str(exc)}

    # --- Basic field statistics ---
    results["field_stats"] = {
        "min": float(np.nanmin(field_2d)),
        "max": float(np.nanmax(field_2d)),
        "mean": float(np.nanmean(field_2d)),
        "std": float(np.nanstd(field_2d)),
        "shape": list(field_2d.shape),
    }

    return results


def analyze_temporal(field_seq: np.ndarray, metadata: dict) -> dict:
    """Run attractor / dynamical-systems analysis on temporal field data."""
    results = {}
    t, h, w = field_seq.shape
    logger.info(
        "Temporal analysis on %d frames of %dx%d fields", t, h, w
    )

    # --- Construct a low-dimensional trajectory ---
    # Use spatial mean voltage + spatial std as a simple 2-D trajectory,
    # plus a few spatial moments for richer embedding.
    mean_v = np.array([np.nanmean(field_seq[i]) for i in range(t)])
    std_v = np.array([np.nanstd(field_seq[i]) for i in range(t)])
    skew_v = np.array([
        float(np.nanmean(((field_seq[i] - np.nanmean(field_seq[i])) / max(np.nanstd(field_seq[i]), 1e-12)) ** 3))
        for i in range(t)
    ])
    trajectory = np.column_stack([mean_v, std_v, skew_v])
    results["trajectory_shape"] = list(trajectory.shape)
    results["mean_vmem_range"] = [float(mean_v.min()), float(mean_v.max())]
    results["std_vmem_range"] = [float(std_v.min()), float(std_v.max())]

    # --- Time-delay embedding ---
    try:
        embedded = embed_trajectory(mean_v, embedding_dimension=3, time_delay=1)
        results["embedding"] = {
            "shape": list(embedded.shape),
            "method": "time_delay",
            "dim": 3,
            "delay": 1,
        }
        logger.info("  Embedded trajectory shape: %s", embedded.shape)
    except Exception as exc:
        logger.warning("  Embedding failed: %s", exc)
        results["embedding"] = {"error": str(exc)}

    # --- Lyapunov spectrum ---
    try:
        spectrum = compute_lyapunov_spectrum(
            trajectory, dt=1.0, n_neighbors=min(10, t - 1)
        )
        attractor_type = classify_attractor_by_lyapunov(spectrum)
        d_ky = kaplan_yorke_dimension(spectrum)
        results["lyapunov"] = {
            "spectrum": [float(s) for s in spectrum],
            "max_exponent": float(spectrum[0]),
            "attractor_type": attractor_type,
            "kaplan_yorke_dimension": float(d_ky),
        }
        logger.info(
            "  Lyapunov: max=%.4f, type=%s, D_KY=%.3f",
            spectrum[0], attractor_type, d_ky,
        )
    except Exception as exc:
        logger.warning("  Lyapunov analysis failed: %s", exc)
        results["lyapunov"] = {"error": str(exc)}

    # --- Recurrence analysis ---
    try:
        ra = RecurrenceAnalysis(threshold=0.1)
        rec_matrix = ra.compute_recurrence_matrix(trajectory)
        rec_rate = float(np.sum(rec_matrix)) / rec_matrix.size
        results["recurrence"] = {
            "matrix_shape": list(rec_matrix.shape),
            "recurrence_rate": rec_rate,
        }
        logger.info("  Recurrence rate: %.4f", rec_rate)
    except Exception as exc:
        logger.warning("  Recurrence analysis failed: %s", exc)
        results["recurrence"] = {"error": str(exc)}

    # --- Topology evolution ---
    # Track how topology changes across timesteps
    try:
        topo_evolution = []
        ph = PersistentHomology(max_dimension=1, persistence_threshold=0.01)
        sample_indices = np.linspace(0, t - 1, min(t, 10), dtype=int)
        for idx in sample_indices:
            diagrams = ph.compute_persistence(field_seq[idx])
            n_feat = sum(len(d.points) for d in diagrams) if diagrams else 0
            topo_evolution.append({
                "timestep": int(idx),
                "total_features": n_feat,
            })
        results["topology_evolution"] = topo_evolution
        logger.info(
            "  Topology evolution tracked across %d sample frames",
            len(sample_indices),
        )
    except Exception as exc:
        logger.warning("  Topology evolution failed: %s", exc)
        results["topology_evolution"] = {"error": str(exc)}

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze BETSE bioelectric data with Mneme"
    )
    parser.add_argument(
        "vmem_dir",
        type=str,
        help="Directory containing Vmem2D_*.csv files from BETSE",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="Grid resolution for field interpolation (default: 64)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/betse_analysis",
        help="Output directory for results (default: results/betse_analysis)",
    )
    parser.add_argument(
        "--exported-data",
        type=str,
        default=None,
        help="Optional path to ExportedData.csv for single-cell analysis",
    )
    args = parser.parse_args()

    start = time.time()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolution = (args.resolution, args.resolution)

    # ----------------------------------------------------------------
    # 1. Load BETSE data
    # ----------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("BETSE -> Mneme Analysis Pipeline")
    logger.info("=" * 60)
    logger.info("Source: %s", args.vmem_dir)
    logger.info("Resolution: %s", resolution)

    field_sequence, metadata = load_betse_timeseries(
        args.vmem_dir, resolution=resolution
    )
    logger.info(
        "Loaded %d frames, %d cells, grid %s",
        metadata["n_timesteps"],
        metadata["n_cells"],
        resolution,
    )
    logger.info(
        "Vmem range: [%.2f, %.2f] mV",
        field_sequence.min(),
        field_sequence.max(),
    )

    # Save the interpolated field as NPZ for future use
    npz_path = output_dir / "betse_field_sequence.npz"
    np.savez_compressed(
        npz_path,
        field_sequence=field_sequence,
        grid_x=metadata["grid_x"],
        grid_y=metadata["grid_y"],
    )
    logger.info("Saved interpolated field to %s", npz_path)

    all_results = {
        "metadata": {
            k: v
            for k, v in metadata.items()
            if not isinstance(v, np.ndarray)
        },
    }

    # ----------------------------------------------------------------
    # 2. Analyze individual frames
    # ----------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("Analyzing individual frames...")
    frame_results = []

    # Analyze first, middle, and last frames
    n = field_sequence.shape[0]
    key_frames = {
        "first": 0,
        "middle": n // 2,
        "last": n - 1,
    }
    for label, idx in key_frames.items():
        logger.info("Frame %d (%s):", idx, label)
        fr = analyze_single_frame(field_sequence[idx], label=f"{label}_t{idx}")
        frame_results.append(fr)

    all_results["frame_analysis"] = frame_results

    # ----------------------------------------------------------------
    # 3. Temporal / dynamical analysis
    # ----------------------------------------------------------------
    if field_sequence.shape[0] > 1:
        logger.info("-" * 60)
        logger.info("Temporal dynamics analysis...")
        temporal_results = analyze_temporal(field_sequence, metadata)
        all_results["temporal_analysis"] = temporal_results

    # ----------------------------------------------------------------
    # 4. Single-cell time series (if provided)
    # ----------------------------------------------------------------
    if args.exported_data:
        logger.info("-" * 60)
        logger.info("Single-cell time series analysis...")
        try:
            ts_data, columns = load_betse_exported_data(args.exported_data)
            logger.info(
                "Loaded %d timesteps, %d columns", ts_data.shape[0], ts_data.shape[1]
            )

            # Extract Vmem column
            vmem_idx = next(
                (i for i, c in enumerate(columns) if "Vmem" in c and "Goldman" not in c),
                1,
            )
            vmem_ts = ts_data[:, vmem_idx]

            try:
                spectrum = compute_lyapunov_spectrum(
                    vmem_ts.reshape(-1, 1), dt=1.0
                )
                atype = classify_attractor_by_lyapunov(spectrum)
                d_ky = kaplan_yorke_dimension(spectrum)
                all_results["single_cell"] = {
                    "n_timesteps": ts_data.shape[0],
                    "columns": columns,
                    "vmem_range": [float(vmem_ts.min()), float(vmem_ts.max())],
                    "vmem_mean": float(vmem_ts.mean()),
                    "lyapunov_spectrum": [float(s) for s in spectrum],
                    "attractor_type": atype,
                    "kaplan_yorke_dim": float(d_ky),
                }
                logger.info(
                    "  Single-cell Lyapunov: max=%.4f, type=%s",
                    spectrum[0], atype,
                )
            except Exception as exc:
                logger.warning("  Single-cell Lyapunov failed: %s", exc)
                all_results["single_cell"] = {
                    "n_timesteps": ts_data.shape[0],
                    "columns": columns,
                    "error": str(exc),
                }
        except Exception as exc:
            logger.warning("Failed to load ExportedData: %s", exc)

    # ----------------------------------------------------------------
    # 5. Save results
    # ----------------------------------------------------------------
    elapsed = time.time() - start
    all_results["execution_time_seconds"] = elapsed

    # Save as JSON for readability
    import json

    json_path = output_dir / "analysis_results.json"

    def _default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return str(obj)

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_default)

    logger.info("-" * 60)
    logger.info("Analysis complete in %.1f seconds", elapsed)
    logger.info("Results saved to: %s", output_dir)
    logger.info("  - %s (interpolated field data)", npz_path.name)
    logger.info("  - %s (analysis results)", json_path.name)

    # ----------------------------------------------------------------
    # 6. Print summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BETSE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Source:      {metadata['source_dir']}")
    print(f"Cells:       {metadata['n_cells']}")
    print(f"Timesteps:   {metadata['n_timesteps']}")
    print(f"Grid:        {resolution[0]}x{resolution[1]}")
    print(f"Vmem range:  [{field_sequence.min():.2f}, {field_sequence.max():.2f}] mV")

    if "temporal_analysis" in all_results:
        ta = all_results["temporal_analysis"]
        if "lyapunov" in ta and "error" not in ta["lyapunov"]:
            ly = ta["lyapunov"]
            print(f"\nDynamics:")
            print(f"  Max Lyapunov:  {ly['max_exponent']:.4f}")
            print(f"  Attractor:     {ly['attractor_type']}")
            print(f"  Kaplan-Yorke:  {ly['kaplan_yorke_dimension']:.3f}")
        if "recurrence" in ta and "error" not in ta["recurrence"]:
            print(f"  Recurrence:    {ta['recurrence']['recurrence_rate']:.4f}")

    for fr in frame_results:
        topo = fr.get("topology", {})
        if "error" not in topo:
            print(f"\nTopology ({fr['label']}):")
            print(f"  Total features: {topo.get('total_features', 'N/A')}")
            for dim in range(3):
                key = f"dim{dim}_count"
                if key in topo:
                    print(
                        f"  H{dim}: {topo[key]} features, "
                        f"max persistence={topo.get(f'dim{dim}_max_persistence', 0):.4f}"
                    )

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
