"""Feature extraction utilities for fields (MVP).

Provides a lightweight `FieldFeatureExtractor` that computes a small set of
interpretable features from 2D fields or 3D sequences.
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np
from scipy.ndimage import gaussian_filter, laplace


class FieldFeatureExtractor:
    """Extract basic features from field data.

    Features (per 2D frame):
    - mean, std, min, max
    - gradient magnitude mean/std
    - Laplacian magnitude mean/std
    - roughness (mean absolute difference between neighboring pixels)
    """

    def __init__(self, smoothing_sigma: float = 0.0) -> None:
        self.smoothing_sigma = float(smoothing_sigma)

    def extract(self, field: np.ndarray) -> Dict[str, Any]:
        """Extract features from a 2D field or 3D sequence (T,H,W).

        Returns a dict with scalar features for 2D, or aggregates for 3D.
        """
        if field.ndim == 2:
            return self._features_2d(field)
        if field.ndim == 3:
            per_frame = [self._features_2d(frame) for frame in field]
            # Aggregate across time
            keys = per_frame[0].keys() if per_frame else []
            agg: Dict[str, Any] = {}
            for k in keys:
                vals = np.array([f[k] for f in per_frame], dtype=float)
                agg[f"{k}_mean_t"] = float(np.mean(vals))
                agg[f"{k}_std_t"] = float(np.std(vals))
            agg["num_frames"] = int(field.shape[0])
            return agg
        raise ValueError("field must be 2D or 3D")

    def _features_2d(self, img: np.ndarray) -> Dict[str, float]:
        if self.smoothing_sigma > 0:
            img = gaussian_filter(img, sigma=self.smoothing_sigma)

        finite = img[np.isfinite(img)]
        if finite.size == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                    "grad_mean": 0.0, "grad_std": 0.0,
                    "lap_mean": 0.0, "lap_std": 0.0, "roughness": 0.0}

        # Basic stats
        mean = float(np.mean(finite))
        std = float(np.std(finite))
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))

        # Gradient magnitude
        gy, gx = np.gradient(img)
        grad_mag = np.sqrt(gx * gx + gy * gy)
        grad_mean = float(np.mean(np.abs(grad_mag)))
        grad_std = float(np.std(grad_mag))

        # Laplacian magnitude
        lap = laplace(img)
        lap_mean = float(np.mean(np.abs(lap)))
        lap_std = float(np.std(lap))

        # Roughness (mean abs difference to neighbors)
        diff_h = np.abs(img[:, 1:] - img[:, :-1])
        diff_v = np.abs(img[1:, :] - img[:-1, :])
        roughness = float(np.mean(np.concatenate([diff_h.flatten(), diff_v.flatten()])))

        return {
            "mean": mean,
            "std": std,
            "min": vmin,
            "max": vmax,
            "grad_mean": grad_mean,
            "grad_std": grad_std,
            "lap_mean": lap_mean,
            "lap_std": lap_std,
            "roughness": roughness,
        }
