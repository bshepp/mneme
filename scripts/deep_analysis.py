#!/usr/bin/env python
"""Deep analysis of BETSE simulation data using Mneme's full toolkit.

Runs PCA-based trajectory extraction, higher-dimensional Lyapunov analysis,
Wasserstein distance tracking, and VAE latent-space embedding on pre-computed
field sequences.

Usage
-----
    python scripts/deep_analysis.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mneme.core.topology import (
    PersistentHomology,
    compute_wasserstein_distance,
    compute_bottleneck_distance,
)
from mneme.core.attractors import (
    compute_lyapunov_spectrum,
    classify_attractor_by_lyapunov,
    kaplan_yorke_dimension,
    RecurrenceAnalysis,
    embed_trajectory,
)
from mneme.models.autoencoders import create_field_vae
from mneme.models.symbolic import SymbolicRegressor, discover_field_dynamics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("deep_analysis")


def load_field_sequence(npz_path: str) -> np.ndarray:
    """Load a pre-computed field sequence from NPZ."""
    data = np.load(npz_path)
    return data["field_sequence"]


# =====================================================================
# 1. PCA / SVD Mode Extraction
# =====================================================================
def pca_trajectory(field_seq: np.ndarray, n_components: int = 10) -> dict:
    """Extract PCA modes from field sequence and build trajectory.

    Flattens each (H, W) frame into a vector, runs SVD to get the
    dominant spatial modes, and returns the time-varying PCA coefficients
    as a high-dimensional trajectory.
    """
    t, h, w = field_seq.shape
    logger.info("PCA: Flattening %d frames of %dx%d to %d-dim vectors", t, h, w, h * w)

    # Flatten: (T, H*W)
    X = field_seq.reshape(t, -1).astype(np.float64)

    # Center
    mean_field = X.mean(axis=0)
    X_centered = X - mean_field

    # SVD (economy)
    n_comp = min(n_components, t, h * w)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Explained variance
    total_var = np.sum(S ** 2)
    explained = (S[:n_comp] ** 2) / total_var
    cumulative = np.cumsum(explained)

    # PCA trajectory: project onto top n_comp modes
    # coefficients[i, j] = how much mode j contributes at timestep i
    coefficients = U[:, :n_comp] * S[:n_comp]  # (T, n_comp)

    logger.info(
        "PCA: Top %d modes explain %.1f%% of variance",
        n_comp,
        cumulative[-1] * 100,
    )
    for i in range(min(5, n_comp)):
        logger.info("  Mode %d: %.2f%% (cumulative %.2f%%)", i, explained[i] * 100, cumulative[i] * 100)

    return {
        "coefficients": coefficients,  # (T, n_comp) - the trajectory
        "singular_values": S[:n_comp].tolist(),
        "explained_variance_ratio": explained.tolist(),
        "cumulative_variance": cumulative.tolist(),
        "n_components": n_comp,
        "total_variance": float(total_var),
    }


# =====================================================================
# 2. Higher-Dimensional Lyapunov Analysis
# =====================================================================
def lyapunov_from_pca(pca_coeffs: np.ndarray, label: str = "") -> dict:
    """Compute Lyapunov spectrum from PCA trajectory."""
    t, d = pca_coeffs.shape
    logger.info("Lyapunov (%s): %d-D trajectory, %d timesteps", label, d, t)

    if t < 100:
        logger.warning("Trajectory too short (%d < 100) for Lyapunov", t)
        return {"error": f"Trajectory too short ({t} points)"}

    try:
        spectrum = compute_lyapunov_spectrum(
            pca_coeffs, dt=1.0, n_neighbors=min(15, t // 10)
        )
        atype = classify_attractor_by_lyapunov(spectrum)
        dky = kaplan_yorke_dimension(spectrum)

        n_positive = int(np.sum(spectrum > 0))
        n_zero = int(np.sum(np.abs(spectrum) < 0.01))
        n_negative = int(np.sum(spectrum < -0.01))

        result = {
            "spectrum": [float(s) for s in spectrum],
            "max_exponent": float(spectrum[0]),
            "min_exponent": float(spectrum[-1]),
            "attractor_type": str(atype),
            "kaplan_yorke_dimension": float(dky),
            "n_positive": n_positive,
            "n_zero": n_zero,
            "n_negative": n_negative,
            "embedding_dim": d,
        }
        logger.info(
            "  Spectrum: max=%.4f, min=%.4f, D_KY=%.3f, type=%s",
            spectrum[0], spectrum[-1], dky, atype,
        )
        logger.info(
            "  Exponent signs: %d+, %d~0, %d-",
            n_positive, n_zero, n_negative,
        )
        return result
    except Exception as exc:
        logger.warning("Lyapunov failed (%s): %s", label, exc)
        return {"error": str(exc)}


# =====================================================================
# 3. Wasserstein Distance Tracking
# =====================================================================
def wasserstein_evolution(field_seq: np.ndarray, n_samples: int = 30) -> dict:
    """Track Wasserstein distance between consecutive frames' persistence diagrams."""
    t = field_seq.shape[0]
    indices = np.linspace(0, t - 1, min(t, n_samples), dtype=int)
    logger.info("Wasserstein: Computing persistence for %d sampled frames", len(indices))

    ph = PersistentHomology(max_dimension=1, persistence_threshold=0.005)

    # Compute all diagrams first
    diagrams_per_frame = []
    for idx in indices:
        try:
            diags = ph.compute_persistence(field_seq[idx])
            diagrams_per_frame.append(diags)
        except Exception as exc:
            logger.warning("  Persistence failed at t=%d: %s", idx, exc)
            diagrams_per_frame.append(None)

    # Compute pairwise consecutive Wasserstein distances
    w_distances = []
    b_distances = []
    for i in range(len(diagrams_per_frame) - 1):
        d1 = diagrams_per_frame[i]
        d2 = diagrams_per_frame[i + 1]

        if d1 is None or d2 is None:
            w_distances.append(None)
            b_distances.append(None)
            continue

        # Match dimension 0 diagrams
        d1_h0 = [d for d in d1 if d.dimension == 0]
        d2_h0 = [d for d in d2 if d.dimension == 0]

        if d1_h0 and d2_h0:
            try:
                wd = compute_wasserstein_distance(d1_h0[0], d2_h0[0], p=2.0)
                bd = compute_bottleneck_distance(d1_h0[0], d2_h0[0])
                w_distances.append(float(wd))
                b_distances.append(float(bd))
            except Exception as exc:
                logger.warning("  Distance computation failed at step %d: %s", i, exc)
                w_distances.append(None)
                b_distances.append(None)
        else:
            w_distances.append(None)
            b_distances.append(None)

    # Also compute distance from first to last (total topological change)
    total_wasserstein = None
    total_bottleneck = None
    if diagrams_per_frame[0] is not None and diagrams_per_frame[-1] is not None:
        d1_h0 = [d for d in diagrams_per_frame[0] if d.dimension == 0]
        d2_h0 = [d for d in diagrams_per_frame[-1] if d.dimension == 0]
        if d1_h0 and d2_h0:
            try:
                total_wasserstein = float(compute_wasserstein_distance(d1_h0[0], d2_h0[0]))
                total_bottleneck = float(compute_bottleneck_distance(d1_h0[0], d2_h0[0]))
            except Exception:
                pass

    # Feature count evolution
    feature_counts = []
    for i, diags in enumerate(diagrams_per_frame):
        if diags is not None:
            n_feat = sum(len(d.points) for d in diags)
            feature_counts.append({"timestep": int(indices[i]), "features": n_feat})

    valid_w = [w for w in w_distances if w is not None]
    valid_b = [b for b in b_distances if b is not None]

    result = {
        "n_samples": len(indices),
        "timesteps": [int(i) for i in indices],
        "wasserstein_consecutive": w_distances,
        "bottleneck_consecutive": b_distances,
        "feature_counts": feature_counts,
        "total_wasserstein_first_to_last": total_wasserstein,
        "total_bottleneck_first_to_last": total_bottleneck,
    }

    if valid_w:
        result["wasserstein_stats"] = {
            "mean": float(np.mean(valid_w)),
            "std": float(np.std(valid_w)),
            "max": float(np.max(valid_w)),
            "min": float(np.min(valid_w)),
        }
        logger.info(
            "  Wasserstein: mean=%.4f, std=%.4f, max=%.4f",
            np.mean(valid_w), np.std(valid_w), np.max(valid_w),
        )
    if valid_b:
        result["bottleneck_stats"] = {
            "mean": float(np.mean(valid_b)),
            "std": float(np.std(valid_b)),
            "max": float(np.max(valid_b)),
            "min": float(np.min(valid_b)),
        }

    if total_wasserstein is not None:
        logger.info("  Total Wasserstein (first->last): %.4f", total_wasserstein)
    if total_bottleneck is not None:
        logger.info("  Total Bottleneck (first->last): %.4f", total_bottleneck)

    return result


# =====================================================================
# 3b. Cross-Frame Wasserstein Distance Matrix
# =====================================================================
def wasserstein_cross_matrix(
    field_seq: np.ndarray,
    n_samples: int = 60,
    max_dimension: int = 1,
) -> dict:
    """Compute NxN Wasserstein distance matrix across sampled frames.

    Parameters
    ----------
    field_seq : np.ndarray
        Field sequence of shape (T, H, W).
    n_samples : int
        Number of frames to sample (evenly spaced).
    max_dimension : int
        Maximum homological dimension for persistence (0 = H0 only, 1 = H0+H1).
    """
    t = field_seq.shape[0]
    indices = np.linspace(0, t - 1, min(t, n_samples), dtype=int)
    n = len(indices)
    logger.info(
        "Wasserstein matrix: %d sampled frames, max_dim=%d", n, max_dimension,
    )

    ph = PersistentHomology(max_dimension=max_dimension, persistence_threshold=0.005)

    # Compute all persistence diagrams
    all_diagrams = []
    for idx in indices:
        try:
            diags = ph.compute_persistence(field_seq[idx])
            all_diagrams.append(diags)
        except Exception as exc:
            logger.warning("  Persistence failed at t=%d: %s", idx, exc)
            all_diagrams.append(None)

    # Log how many dimensions we got
    for i, diags in enumerate(all_diagrams):
        if diags is not None:
            dims_found = [d.dimension for d in diags]
            n_pts = {d.dimension: len(d.points) for d in diags}
            logger.info(
                "  Frame %d (t=%d): dims=%s, points=%s",
                i, indices[i], dims_found, n_pts,
            )
            break

    # Build matrices per homological dimension
    result = {
        "n_samples": n,
        "timesteps": [int(i) for i in indices],
    }

    for dim in range(max_dimension + 1):
        dim_label = f"H{dim}"
        logger.info("  Computing %s Wasserstein matrix (%dx%d)...", dim_label, n, n)

        w_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d1 = all_diagrams[i]
                d2 = all_diagrams[j]
                if d1 is None or d2 is None:
                    w_matrix[i, j] = np.nan
                    w_matrix[j, i] = np.nan
                    continue
                # Get diagrams for this dimension
                d1_dim = [d for d in d1 if d.dimension == dim]
                d2_dim = [d for d in d2 if d.dimension == dim]
                if d1_dim and d2_dim:
                    try:
                        wd = compute_wasserstein_distance(d1_dim[0], d2_dim[0], p=2.0)
                        w_matrix[i, j] = wd
                        w_matrix[j, i] = wd
                    except Exception:
                        w_matrix[i, j] = np.nan
                        w_matrix[j, i] = np.nan
                else:
                    w_matrix[i, j] = np.nan
                    w_matrix[j, i] = np.nan

        # Stats
        upper = w_matrix[np.triu_indices(n, k=1)]
        valid = upper[np.isfinite(upper)]
        if len(valid) > 0:
            result[f"{dim_label}_matrix_stats"] = {
                "mean": float(np.mean(valid)),
                "std": float(np.std(valid)),
                "max": float(np.max(valid)),
                "min": float(np.min(valid)),
                "n_valid_pairs": int(len(valid)),
            }
            logger.info(
                "  %s: mean=%.4f, std=%.4f, max=%.4f, min=%.4f",
                dim_label, np.mean(valid), np.std(valid),
                np.max(valid), np.min(valid),
            )

            # Check for topological phase transitions: large jumps in consecutive W
            consecutive_w = np.array([w_matrix[i, i + 1] for i in range(n - 1)])
            valid_consec = consecutive_w[np.isfinite(consecutive_w)]
            if len(valid_consec) > 2:
                mean_c = np.mean(valid_consec)
                std_c = np.std(valid_consec)
                # Frames where distance is >2 std above mean
                if std_c > 0:
                    jumps = np.where(valid_consec > mean_c + 2 * std_c)[0]
                    if len(jumps) > 0:
                        result[f"{dim_label}_phase_transitions"] = [
                            {"from_frame": int(indices[j]),
                             "to_frame": int(indices[j + 1]),
                             "wasserstein": float(valid_consec[j])}
                            for j in jumps
                        ]
                        logger.info(
                            "  %s phase transitions at frames: %s",
                            dim_label, [int(indices[j]) for j in jumps],
                        )

            # Check for topological recurrence: low W for distant timesteps
            diag_distances = []
            for offset in [n // 4, n // 2, 3 * n // 4]:
                if offset > 0 and offset < n:
                    vals = [w_matrix[i, i + offset]
                            for i in range(n - offset)
                            if np.isfinite(w_matrix[i, i + offset])]
                    if vals:
                        diag_distances.append({
                            "offset_frames": int(offset),
                            "mean_distance": float(np.mean(vals)),
                        })
            if diag_distances:
                result[f"{dim_label}_recurrence_by_offset"] = diag_distances

        # Save matrix as list of lists (JSON-serializable)
        result[f"{dim_label}_matrix"] = w_matrix.tolist()

    return result


# =====================================================================
# 4. Recurrence Analysis with PCA trajectory
# =====================================================================
def recurrence_from_pca(pca_coeffs: np.ndarray, label: str = "") -> dict:
    """Compute recurrence analysis from PCA trajectory."""
    logger.info("Recurrence (%s): %d-D trajectory", label, pca_coeffs.shape[1])
    try:
        # Normalize trajectory for threshold computation
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled = scaler.fit_transform(pca_coeffs)

        # Use 10% of max distance as threshold
        from scipy.spatial.distance import pdist
        distances = pdist(scaled[:200])  # subsample for speed
        threshold = np.percentile(distances, 10)

        ra = RecurrenceAnalysis(threshold=threshold)
        rec_matrix = ra.compute_recurrence_matrix(scaled)
        rec_rate = float(np.sum(rec_matrix)) / rec_matrix.size

        # Diagonal line analysis (determinism)
        # Count diagonal lines of length >= 2
        n = rec_matrix.shape[0]
        diag_lengths = []
        for offset in range(1, min(n, 200)):
            diag = np.diag(rec_matrix, offset)
            # Run-length encoding
            current_len = 0
            for val in diag:
                if val:
                    current_len += 1
                else:
                    if current_len >= 2:
                        diag_lengths.append(current_len)
                    current_len = 0
            if current_len >= 2:
                diag_lengths.append(current_len)

        determinism = sum(diag_lengths) / max(np.sum(rec_matrix), 1)

        result = {
            "recurrence_rate": rec_rate,
            "threshold": float(threshold),
            "determinism": float(determinism),
            "n_diagonal_lines": len(diag_lengths),
            "max_diagonal_length": int(max(diag_lengths)) if diag_lengths else 0,
            "mean_diagonal_length": float(np.mean(diag_lengths)) if diag_lengths else 0,
        }
        logger.info(
            "  Recurrence rate: %.4f, Determinism: %.4f, Max diag: %d",
            rec_rate, determinism, result["max_diagonal_length"],
        )
        return result
    except Exception as exc:
        logger.warning("Recurrence failed (%s): %s", label, exc)
        return {"error": str(exc)}


# =====================================================================
# 5. VAE Latent-Space Analysis
# =====================================================================
def vae_analysis(
    field_sequences: dict,
    latent_dim: int = 16,
    epochs: int = 50,
) -> dict:
    """Train VAE on all field data and extract latent trajectories.

    Parameters
    ----------
    field_sequences : dict
        Mapping of label -> (T, H, W) arrays
    """
    # Combine all sequences for training
    all_fields = []
    seq_boundaries = {}
    offset = 0
    for label, seq in field_sequences.items():
        t = seq.shape[0]
        seq_boundaries[label] = (offset, offset + t)
        all_fields.append(seq)
        offset += t

    combined = np.concatenate(all_fields, axis=0)
    logger.info(
        "VAE: Training on %d total frames from %d sequences, shape per frame: %s",
        combined.shape[0], len(field_sequences), combined.shape[1:],
    )

    # Normalize to [0, 1] for VAE
    vmin, vmax = combined.min(), combined.max()
    combined_norm = (combined - vmin) / (vmax - vmin + 1e-12)

    # Train/val split (80/20)
    n = combined_norm.shape[0]
    indices = np.random.RandomState(42).permutation(n)
    n_train = int(0.8 * n)
    train_data = combined_norm[indices[:n_train]]
    val_data = combined_norm[indices[n_train:]]

    logger.info("VAE: Train=%d, Val=%d, Latent=%d", n_train, n - n_train, latent_dim)

    # Create and train VAE
    h, w = combined.shape[1], combined.shape[2]
    vae = create_field_vae((h, w), latent_dim=latent_dim)

    try:
        train_result = vae.fit(
            train_data,
            val_data=val_data,
            epochs=epochs,
            batch_size=32,
            learning_rate=1e-3,
            early_stopping_patience=10,
            verbose=False,
        )
        logger.info(
            "VAE: Training complete. Best epoch=%d, Final loss=%.4f",
            train_result.best_epoch,
            train_result.final_loss,
        )
    except Exception as exc:
        logger.error("VAE training failed: %s", exc)
        return {"error": str(exc)}

    # Encode ALL frames (in original order)
    latent_all = vae.encode_fields(combined_norm)
    logger.info("VAE: Encoded all frames to shape %s", latent_all.shape)

    # Extract per-sequence latent trajectories
    results = {
        "latent_dim": latent_dim,
        "n_total_frames": n,
        "training": {
            "best_epoch": train_result.best_epoch,
            "final_loss": float(train_result.final_loss),
            "train_losses": [float(l) for l in train_result.train_losses[-5:]],
            "val_losses": [float(l) for l in train_result.val_losses[-5:]] if train_result.val_losses else None,
        },
        "per_sequence": {},
    }

    for label, (start, end) in seq_boundaries.items():
        latent_seq = latent_all[start:end]  # (T, latent_dim)
        logger.info("VAE: Sequence '%s' latent trajectory shape: %s", label, latent_seq.shape)

        seq_result = {
            "latent_trajectory_shape": list(latent_seq.shape),
            "latent_mean": latent_seq.mean(axis=0).tolist(),
            "latent_std": latent_seq.std(axis=0).tolist(),
        }

        # Lyapunov from latent trajectory
        if latent_seq.shape[0] >= 100:
            try:
                spectrum = compute_lyapunov_spectrum(
                    latent_seq, dt=1.0, n_neighbors=min(15, latent_seq.shape[0] // 10)
                )
                atype = classify_attractor_by_lyapunov(spectrum)
                dky = kaplan_yorke_dimension(spectrum)
                n_pos = int(np.sum(spectrum > 0))
                n_neg = int(np.sum(spectrum < -0.01))

                seq_result["lyapunov"] = {
                    "spectrum": [float(s) for s in spectrum],
                    "max_exponent": float(spectrum[0]),
                    "min_exponent": float(spectrum[-1]),
                    "attractor_type": str(atype),
                    "kaplan_yorke_dimension": float(dky),
                    "n_positive": n_pos,
                    "n_negative": n_neg,
                }
                logger.info(
                    "  %s Lyapunov (VAE): max=%.4f, D_KY=%.3f, type=%s, %d+/%d-",
                    label, spectrum[0], dky, atype, n_pos, n_neg,
                )
            except Exception as exc:
                logger.warning("  %s Lyapunov (VAE) failed: %s", label, exc)
                seq_result["lyapunov"] = {"error": str(exc)}

        # Recurrence from latent trajectory
        try:
            from sklearn.preprocessing import StandardScaler
            scaled = StandardScaler().fit_transform(latent_seq)
            ra = RecurrenceAnalysis(threshold=0.5)
            rec_matrix = ra.compute_recurrence_matrix(scaled)
            rec_rate = float(np.sum(rec_matrix)) / rec_matrix.size
            seq_result["recurrence_rate"] = rec_rate
            logger.info("  %s Recurrence (VAE): %.4f", label, rec_rate)
        except Exception as exc:
            seq_result["recurrence_rate"] = {"error": str(exc)}

        results["per_sequence"][label] = seq_result

    return results


# =====================================================================
# 6. Symbolic Regression on PCA Mode Dynamics
# =====================================================================
def symbolic_regression_pca_modes(
    pca_coefficients: np.ndarray,
    label: str = "",
    niterations: int = 50,
) -> dict:
    """Discover ODEs governing PCA mode evolution: dc_i/dt = f(c_1,...,c_n).

    Parameters
    ----------
    pca_coefficients : np.ndarray
        PCA trajectory of shape (T, n_modes).
    label : str
        Dataset label for logging.
    niterations : int
        Number of PySR iterations.
    """
    t, n_modes = pca_coefficients.shape
    logger.info(
        "Symbolic regression (%s): %d modes, %d timesteps, %d iterations",
        label, n_modes, t, niterations,
    )

    # Compute time derivatives: dc/dt from consecutive timesteps
    dc_dt = np.diff(pca_coefficients, axis=0)  # (T-1, n_modes)
    X = pca_coefficients[:-1]  # features: c_1..c_n at time t

    variable_names = [f"c{i}" for i in range(n_modes)]

    mode_results = {}

    for mode_idx in range(n_modes):
        y = dc_dt[:, mode_idx]  # target: dc_i/dt

        logger.info("  Mode %d (dc%d/dt):", mode_idx, mode_idx)

        try:
            sr = SymbolicRegressor(
                niterations=niterations,
                progress=False,
                random_state=42,
                complexity_penalty=0.003,
                max_complexity=20,
            )
            sr.fit(X, y, variable_names=variable_names)

            best_eq = sr.get_best_equation()
            r2 = sr.score(X, y)
            all_eqs = sr.get_equations(n_best=3)

            mode_results[f"mode_{mode_idx}"] = {
                "best_equation": best_eq,
                "r2_score": float(r2),
                "top_equations": all_eqs,
                "using_pysr": sr._using_pysr,
                "n_samples": int(len(y)),
            }

            logger.info("    Best: dc%d/dt = %s", mode_idx, best_eq)
            logger.info("    R² = %.4f (PySR=%s)", r2, sr._using_pysr)

        except Exception as exc:
            logger.warning("    Mode %d failed: %s", mode_idx, exc)
            mode_results[f"mode_{mode_idx}"] = {"error": str(exc)}

    return {
        "label": label,
        "n_modes": n_modes,
        "n_timesteps": t,
        "niterations": niterations,
        "mode_equations": mode_results,
    }


def symbolic_regression_field(
    field_seq: np.ndarray,
    label: str = "",
    niterations: int = 50,
) -> dict:
    """Run spatial PDE discovery on raw field sequence.

    Uses the existing discover_field_dynamics() convenience function.
    """
    logger.info(
        "Spatial symbolic regression (%s): shape=%s, %d iterations",
        label, field_seq.shape, niterations,
    )
    try:
        result = discover_field_dynamics(
            field_seq,
            dt=1.0,
            spatial_features=True,
            niterations=niterations,
            progress=False,
            random_state=42,
        )
        return {
            "best_equation": result["best_equation"],
            "equations": result["equations"][:3],
            "r2_score": float(result["r2_score"]),
            "features_used": result["features_used"],
        }
    except Exception as exc:
        logger.warning("Spatial symbolic regression failed (%s): %s", label, exc)
        return {"error": str(exc)}


# =====================================================================
# Main
# =====================================================================
def analyze_dataset(label: str, npz_path: str) -> dict:
    """Run full deep analysis on one dataset."""
    logger.info("=" * 70)
    logger.info("DEEP ANALYSIS: %s", label)
    logger.info("=" * 70)

    field_seq = load_field_sequence(npz_path)
    t, h, w = field_seq.shape
    logger.info("Loaded: %d frames of %dx%d", t, h, w)

    result = {"label": label, "shape": list(field_seq.shape)}

    # --- PCA ---
    # Use 5 components for patterns (effective rank ~5), 10 for others
    n_pca = 5 if label == "patterns" else 10
    logger.info("-" * 60)
    logger.info("Phase 1: PCA Mode Extraction (n_components=%d)", n_pca)
    pca_result = pca_trajectory(field_seq, n_components=n_pca)
    result["pca"] = {k: v for k, v in pca_result.items() if k != "coefficients"}
    # Store coefficients for later use (symbolic regression) - not serialized
    result["_pca_coefficients"] = pca_result["coefficients"]

    # --- Lyapunov from PCA ---
    logger.info("-" * 60)
    logger.info("Phase 2: Lyapunov from PCA trajectory (%d-D)", pca_result["n_components"])
    lyap_result = lyapunov_from_pca(pca_result["coefficients"], label=label)
    result["lyapunov_pca"] = lyap_result

    # --- Recurrence from PCA ---
    logger.info("-" * 60)
    logger.info("Phase 3: Recurrence from PCA trajectory")
    rec_result = recurrence_from_pca(pca_result["coefficients"], label=label)
    result["recurrence_pca"] = rec_result

    # --- Wasserstein ---
    logger.info("-" * 60)
    logger.info("Phase 4: Wasserstein distance evolution")
    wass_result = wasserstein_evolution(field_seq, n_samples=30)
    result["wasserstein"] = wass_result

    return result, field_seq


def main():
    start = time.time()

    # Define datasets -- now including patterns
    results_dir = PROJECT_ROOT / "results"
    datasets = {
        "attractors_1_sim1": results_dir / "attractors_1_sim1" / "betse_field_sequence.npz",
        "attractors_1_sim2": results_dir / "attractors_1_sim2" / "betse_field_sequence.npz",
        "physiology": results_dir / "physiology_sim1" / "betse_field_sequence.npz",
        "patterns": results_dir / "patterns_sim" / "betse_field_sequence.npz",
    }

    # Check what exists
    available = {}
    for label, path in datasets.items():
        if path.exists():
            available[label] = str(path)
            logger.info("Found: %s -> %s", label, path)
        else:
            logger.warning("Missing: %s -> %s", label, path)

    if not available:
        logger.error("No datasets found!")
        return

    # Run per-dataset analysis
    all_results = {}
    field_sequences = {}
    pca_trajectories = {}  # Store PCA coefficients for symbolic regression

    for label, npz_path in available.items():
        result, field_seq = analyze_dataset(label, npz_path)
        all_results[label] = result
        field_sequences[label] = field_seq
        # Store PCA coefficients for later use
        pca_trajectories[label] = result.get("_pca_coefficients")

    # --- Cross-frame Wasserstein Matrix (Task 5) ---
    # Run on patterns dataset with GUDHI-powered H0+H1
    if "patterns" in field_sequences:
        logger.info("=" * 70)
        logger.info("Phase 5a: Cross-Frame Wasserstein Matrix (patterns)")
        logger.info("=" * 70)
        wmatrix_result = wasserstein_cross_matrix(
            field_sequences["patterns"],
            n_samples=60,
            max_dimension=1,  # H0 + H1 with GUDHI
        )
        all_results["wasserstein_cross_matrix_patterns"] = wmatrix_result

        # Save matrix as NPZ separately (too large for JSON)
        output_dir = results_dir / "deep_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        matrix_data = {}
        for key in ["H0_matrix", "H1_matrix"]:
            if key in wmatrix_result:
                matrix_data[key] = np.array(wmatrix_result[key])
        if matrix_data:
            npz_path = output_dir / "wasserstein_cross_matrix_patterns.npz"
            np.savez_compressed(npz_path, **matrix_data,
                                timesteps=np.array(wmatrix_result["timesteps"]))
            logger.info("Saved Wasserstein matrix to %s", npz_path)
            # Remove raw matrices from JSON (too large)
            for key in ["H0_matrix", "H1_matrix"]:
                if key in wmatrix_result:
                    del wmatrix_result[key]

    # --- VAE Analysis (trained on all data jointly) ---
    logger.info("=" * 70)
    logger.info("Phase 5b: VAE Latent-Space Analysis")
    logger.info("=" * 70)
    vae_result = vae_analysis(field_sequences, latent_dim=16, epochs=50)
    all_results["vae_analysis"] = vae_result

    # --- Symbolic Regression on PCA Modes (Task 2) ---
    # Run on patterns (effective rank 5) and optionally on others
    if "patterns" in available and pca_trajectories.get("patterns") is not None:
        logger.info("=" * 70)
        logger.info("Phase 6: Symbolic Regression on PCA Modes (patterns)")
        logger.info("=" * 70)
        pca_coeffs = pca_trajectories["patterns"]
        # Use only the effective 5 modes for patterns
        n_eff = min(5, pca_coeffs.shape[1])
        sr_pca_result = symbolic_regression_pca_modes(
            pca_coeffs[:, :n_eff],
            label="patterns",
            niterations=50,
        )
        all_results["symbolic_regression_pca_patterns"] = sr_pca_result

        # Also run spatial PDE discovery on the raw patterns field
        logger.info("=" * 70)
        logger.info("Phase 6b: Spatial PDE Discovery (patterns)")
        logger.info("=" * 70)
        sr_field_result = symbolic_regression_field(
            field_sequences["patterns"],
            label="patterns",
            niterations=50,
        )
        all_results["symbolic_regression_field_patterns"] = sr_field_result

    # --- Save ---
    elapsed = time.time() - start
    all_results["execution_time_seconds"] = elapsed

    output_dir = results_dir / "deep_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    def _default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return str(obj)

    # Strip non-serializable internal keys before saving
    for label_key in list(all_results.keys()):
        if isinstance(all_results[label_key], dict):
            all_results[label_key].pop("_pca_coefficients", None)

    json_path = output_dir / "deep_analysis_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_default)

    # --- Print Summary ---
    print("\n" + "=" * 70)
    print("DEEP ANALYSIS SUMMARY")
    print("=" * 70)

    for label in available:
        r = all_results[label]
        print(f"\n{'-' * 60}")
        print(f"  {label}")
        print(f"{'-' * 60}")

        # PCA
        pca = r["pca"]
        print(f"  PCA: {pca['n_components']} modes, top-1 explains {pca['explained_variance_ratio'][0]*100:.1f}%")
        print(f"  PCA: cumulative top-5 = {pca['cumulative_variance'][min(4, len(pca['cumulative_variance'])-1)]*100:.1f}%")

        # Lyapunov
        ly = r.get("lyapunov_pca", {})
        if "error" not in ly:
            print(f"  Lyapunov (PCA {ly.get('embedding_dim', '?')}D):")
            print(f"    max={ly['max_exponent']:.4f}, min={ly['min_exponent']:.4f}")
            print(f"    D_KY={ly['kaplan_yorke_dimension']:.3f}, type={ly['attractor_type']}")
            print(f"    signs: {ly['n_positive']}+, {ly['n_zero']}~0, {ly['n_negative']}-")
        else:
            print(f"  Lyapunov: {ly['error']}")

        # Recurrence
        rec = r.get("recurrence_pca", {})
        if "error" not in rec:
            print(f"  Recurrence (PCA): rate={rec['recurrence_rate']:.4f}, det={rec['determinism']:.4f}")
            print(f"    max_diag={rec['max_diagonal_length']}, mean_diag={rec['mean_diagonal_length']:.1f}")

        # Wasserstein
        ws = r.get("wasserstein", {})
        wstats = ws.get("wasserstein_stats", {})
        if wstats:
            print(f"  Wasserstein (H0): mean={wstats['mean']:.4f}, std={wstats['std']:.4f}, max={wstats['max']:.4f}")
        if ws.get("total_wasserstein_first_to_last"):
            print(f"  Total topological drift: W={ws['total_wasserstein_first_to_last']:.4f}, B={ws['total_bottleneck_first_to_last']:.4f}")

    # Wasserstein matrix summary
    wm = all_results.get("wasserstein_cross_matrix_patterns", {})
    if wm:
        print(f"\n{'-' * 60}")
        print(f"  Cross-Frame Wasserstein Matrix (patterns, {wm.get('n_samples', '?')} frames)")
        print(f"{'-' * 60}")
        for dim_label in ["H0", "H1"]:
            stats = wm.get(f"{dim_label}_matrix_stats", {})
            if stats:
                print(f"  {dim_label}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                      f"max={stats['max']:.4f}, min={stats['min']:.4f}")
            transitions = wm.get(f"{dim_label}_phase_transitions", [])
            if transitions:
                print(f"  {dim_label} phase transitions: {len(transitions)} detected")

    # Symbolic regression summary
    sr_pca = all_results.get("symbolic_regression_pca_patterns", {})
    if sr_pca and "mode_equations" in sr_pca:
        print(f"\n{'-' * 60}")
        print(f"  Symbolic Regression: PCA Mode ODEs (patterns)")
        print(f"{'-' * 60}")
        for mode_key, mode_data in sr_pca["mode_equations"].items():
            if "error" not in mode_data:
                print(f"  {mode_key}: dc/dt = {mode_data['best_equation']}")
                print(f"    R² = {mode_data['r2_score']:.4f} (PySR={mode_data.get('using_pysr', '?')})")

    sr_field = all_results.get("symbolic_regression_field_patterns", {})
    if sr_field and "error" not in sr_field:
        print(f"\n  Spatial PDE: du/dt = {sr_field['best_equation']}")
        print(f"    R² = {sr_field['r2_score']:.4f}")

    # VAE summary
    vae = all_results.get("vae_analysis", {})
    if "error" not in vae:
        print(f"\n{'-' * 60}")
        print(f"  VAE (latent_dim={vae.get('latent_dim', '?')}, {vae.get('n_total_frames', '?')} frames)")
        print(f"{'-' * 60}")
        tr = vae.get("training", {})
        print(f"  Training: best_epoch={tr.get('best_epoch')}, loss={tr.get('final_loss', 0):.4f}")

        for label, seq_r in vae.get("per_sequence", {}).items():
            ly = seq_r.get("lyapunov", {})
            if "error" not in ly:
                print(f"  {label} (VAE): max_lyap={ly['max_exponent']:.4f}, D_KY={ly['kaplan_yorke_dimension']:.3f}, type={ly['attractor_type']}")
                print(f"    signs: {ly['n_positive']}+/{ly['n_negative']}-")
            rr = seq_r.get("recurrence_rate")
            if isinstance(rr, float):
                print(f"    recurrence={rr:.4f}")

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results: {json_path}")


if __name__ == "__main__":
    main()
