"""Support status of Mneme components.

Mneme's components fall into three tiers (see docs/SCOPE.md):

- core: tested against known answers and supported.
- frozen: usable within a documented operating range, not under
  development.
- experimental: no validation against known answers. Output should not be
  used as evidence for a scientific claim.

Experimental components emit :class:`ExperimentalWarning` when constructed.
"""

import warnings

__all__ = ["ExperimentalWarning", "warn_experimental"]


class ExperimentalWarning(UserWarning):
    """Raised when an unvalidated component is used."""


def warn_experimental(component: str, stacklevel: int = 3) -> None:
    """Warn that `component` has not been validated against known answers."""
    warnings.warn(
        f"{component} is experimental: it has not been validated against "
        "known answers, and its output should not be used as evidence for a "
        "scientific claim. See docs/SCOPE.md.",
        ExperimentalWarning,
        stacklevel=stacklevel,
    )
