"""Static-only faithful adaptation of the original DySurv benchmark model.

The six static benchmark notebooks under ``src/models_references/DySurv`` use
the same MLP-VAE structure: F -> 3F -> 5F -> 3F -> latent, a static decoder,
and a LogisticHazard survival head. This module keeps that structure while
making evaluation deterministic and exposing decoder activation explicitly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _hidden_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    dropout: float,
    activation: str = "relu",
) -> nn.Sequential:
    modules: list[nn.Module] = []
    previous = int(input_dim)
    for hidden in hidden_dims:
        modules.append(nn.Linear(previous, int(hidden)))
        if activation == "relu":
            modules.append(nn.ReLU())
        elif activation != "none":
            raise ValueError(f"Unsupported activation: {activation}")
        if dropout > 0:
            modules.append(nn.Dropout(float(dropout)))
        previous = int(hidden)
    modules.append(nn.Linear(previous, int(output_dim)))
    return nn.Sequential(*modules)


class DySurvStaticFaithful72h(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 20,
        encoder_multiplier: list[int] | None = None,
        decoder_multiplier: list[int] | None = None,
        survival_multiplier: list[int] | None = None,
        decoder_activation: str = "none",
        dropout: float = 0.0,
        num_durations: int = 10,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.num_durations = int(num_durations)
        encoder_multiplier = encoder_multiplier or [3, 5, 3]
        decoder_multiplier = decoder_multiplier or [3, 5, 3]
        survival_multiplier = survival_multiplier or [3, 5, 3]
        encoder_dims = [int(multiplier) * self.input_dim for multiplier in encoder_multiplier]
        decoder_dims = [int(multiplier) * self.input_dim for multiplier in decoder_multiplier]
        survival_dims = [int(multiplier) * self.input_dim for multiplier in survival_multiplier]

        self.encoder_body = _hidden_mlp(
            self.input_dim,
            encoder_dims[:-1],
            encoder_dims[-1],
            dropout,
            activation="relu",
        )
        self.mu = nn.Linear(encoder_dims[-1], self.latent_dim)
        self.logvar = nn.Linear(encoder_dims[-1], self.latent_dim)
        self.decoder = _hidden_mlp(
            self.latent_dim,
            decoder_dims,
            self.input_dim,
            dropout,
            activation=decoder_activation,
        )
        self.survival_head = _hidden_mlp(
            self.latent_dim,
            survival_dims,
            self.num_durations,
            dropout,
            activation="relu",
        )

    def encode(self, x_static: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder_body(x_static)
        return self.mu(encoded), self.logvar(encoded)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x_static: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x_static)
        z = self.reparameterize(mu, logvar)
        return {
            "reconstruction": self.decoder(z),
            "logits": self.survival_head(z),
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    def predict_logits(self, x_static: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x_static)
        return self.survival_head(mu)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))
