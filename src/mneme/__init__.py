"""
Mneme: Detecting field-like memory structures in biological systems.

An exploratory research system designed to detect emergent memory structures
embedded in biological systems, beginning with planarian regeneration and
bioelectric data.
"""

__version__ = "0.1.0"
__author__ = "Mneme Development Team"

from . import core
from . import models
from . import data
from . import analysis
from . import utils

__all__ = ["core", "models", "data", "analysis", "utils"]