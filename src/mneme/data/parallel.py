"""Parallel pipeline runner (MVP).

Provides a thin wrapper to run a pipeline over a list of files in parallel.
"""
from __future__ import annotations

from typing import List, Any, Dict, Optional, Union
from pathlib import Path
import multiprocessing as mp
import traceback
import numpy as np

from .loaders import create_data_loader


class ParallelPipeline:
    def __init__(self, pipeline, backend: str = 'multiprocessing', n_workers: int = 4):
        self.pipeline = pipeline
        self.backend = backend
        self.n_workers = max(1, int(n_workers))

    def _process_path(self, path: Union[str, Path]) -> Dict[str, Any]:
        path = Path(path)
        try:
            if path.is_dir():
                loader = create_data_loader(path)
                data = next(iter(loader))
            else:
                if path.suffix == '.npz':
                    loaded = np.load(path)
                    data = loaded['data']
                elif path.suffix == '.npy':
                    data = np.load(path)
                else:
                    raise ValueError(f"Unsupported file format: {path.suffix}")
            result = self.pipeline.run(data)
            return {'path': str(path), 'result': result}
        except Exception:
            return {'path': str(path), 'error': traceback.format_exc()}

    def map(self, files: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        if self.n_workers == 1 or len(files) == 1:
            return [self._process_path(p) for p in files]
        with mp.Pool(processes=self.n_workers) as pool:
            return pool.map(self._process_path, files)
