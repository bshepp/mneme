"""Pipeline monitoring utilities (MVP).

Provides a simple monitor for timing stages and tracking errors.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Any


class PipelineMonitor:
    def __init__(self) -> None:
        self._start = None
        self._durations: Dict[str, float] = {}
        self._errors: Dict[str, int] = {}

    def start(self) -> None:
        self._start = time.time()

    @contextmanager
    def track_stage(self, name: str):
        t0 = time.time()
        try:
            yield
        except Exception:
            self._errors[name] = self._errors.get(name, 0) + 1
            raise
        finally:
            self._durations[name] = self._durations.get(name, 0.0) + (time.time() - t0)

    def get_metrics(self) -> Dict[str, Any]:
        total = None
        if self._start is not None:
            total = time.time() - self._start
        return {
            'total_time_s': total,
            'stage_durations_s': dict(self._durations),
            'errors': dict(self._errors),
        }
