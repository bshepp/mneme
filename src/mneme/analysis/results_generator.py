"""Result generation utilities (alignment with docs).

Provides a simple facade that compiles disparate outputs into a dictionary and
saves them using mneme.utils.io.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path

from .results import ResultManager
from ..utils.io import save_results


class ResultGenerator:
    """Compile and save analysis results.

    This class is intentionally simple for MVP and uses existing IO utilities.
    """

    def __init__(self, base_dir: Optional[str] = None) -> None:
        self.base_dir = base_dir
        self._manager = ResultManager(base_dir) if base_dir else None

    def compile(self, parts: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a bundle of parts into a serializable dictionary.

        Keys may include: 'raw_data', 'processed_data', 'reconstruction',
        'topology', 'attractors', 'latent_space', 'metadata'.
        """
        bundle: Dict[str, Any] = {}
        for k in [
            'raw_data', 'processed_data', 'reconstruction', 'topology',
            'attractors', 'latent_space', 'metadata'
        ]:
            if k in parts and parts[k] is not None:
                bundle[k] = parts[k]
        return bundle

    def save(self, bundle: Dict[str, Any], output_path: str | Path, format: str = 'hdf5') -> Path:
        """Save bundle using existing IO."""
        save_results(bundle, output_path, format=format)
        return Path(output_path)
