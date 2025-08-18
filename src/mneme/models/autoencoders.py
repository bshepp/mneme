"""Placeholder autoencoder models.

This module provides a minimal, typed placeholder implementation for
field autoencoders so that imports and basic interfaces work.
"""
from __future__ import annotations

from typing import Tuple

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - allow import without torch in CI
    torch = None  # type: ignore
    nn = None  # type: ignore


class FieldAutoencoder(nn.Module if nn is not None else object):
    """Variational autoencoder placeholder for field data.

    Parameters
    ----------
    input_shape: Tuple[int, int]
        Shape of the 2D field (H, W)
    latent_dim: int
        Latent dimension size
    architecture: str
        Architecture type (ignored in placeholder)
    """

    def __init__(self, input_shape: Tuple[int, int], latent_dim: int = 32, architecture: str = "convolutional") -> None:
        if nn is None:
            # Lightweight placeholder with no torch dependency
            self.input_shape = input_shape
            self.latent_dim = latent_dim
            self.architecture = architecture
            return

        super().__init__()
        c = 1
        h, w = input_shape
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * h * w, 128),
            nn.ReLU(),
        )
        self.mu = nn.Linear(128, latent_dim)
        self.log_var = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, c * h * w),
        )
        self._h = h
        self._w = w

    def encode(self, field):  # type: ignore[no-untyped-def]
        if nn is None:
            return None, None  # placeholder
        h = self.encoder(field)
        return self.mu(h), self.log_var(h)

    def decode(self, z):  # type: ignore[no-untyped-def]
        if nn is None:
            return None
        x = self.decoder(z)
        return x.view(-1, 1, self._h, self._w)

    def forward(self, field):  # type: ignore[no-untyped-def]
        if nn is None:
            return None, None, None
        mu, log_var = self.encode(field)
        std = (0.5 * log_var).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon = self.decode(z)
        return recon, mu, log_var
