from __future__ import annotations

import torch
from torch import nn


class VAE(nn.Module):
    """Small configurable VAE for the 8-dimensional IDS feature vector."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 16,
        latent_dim: int = 4,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor):
        hidden = self.encoder(x)
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        return mu, logvar

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar, z

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic reconstruction using the latent mean."""
        mu, _ = self.encode(x)
        return self.decode(mu)


def vae_loss(
    reconstruction: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
):
    """Return total VAE loss and its two components."""

    reconstruction_loss = nn.functional.mse_loss(
        reconstruction,
        x,
        reduction="mean",
    )

    kl_loss = -0.5 * torch.mean(
        1.0 + logvar - mu.pow(2) - logvar.exp()
    )

    total_loss = reconstruction_loss + beta * kl_loss

    return total_loss, reconstruction_loss, kl_loss
