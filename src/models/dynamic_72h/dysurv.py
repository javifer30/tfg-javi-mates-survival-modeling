"""DySurv adaptation for the TFG dynamic_72h experiment.

Reference reviewed:
src/models_references/DySurv/Models/Results/DySurv.ipynb

Replicated components:
- LSTM encoder over the 72-hour trajectory.
- MLP encoder producing mu/logvar.
- Reparameterization trick for latent z.
- Survival head over discrete daily hazards.
- Decoder reconstructing the temporal trajectory from z.
- Weighted survival, reconstruction and KL losses.

Adaptation:
- The original notebook appends the target duration to the input and conditions
  the decoder on it. Here the target is never used as model input, so the
  survival branch cannot receive post-outcome information.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def mlp(input_dim: int, layers: list[int], output_dim: int, dropout: float) -> nn.Sequential:
    modules = []
    prev = input_dim
    for hidden in layers:
        modules.extend([nn.Linear(prev, hidden), nn.ReLU()])
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        prev = hidden
    modules.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*modules)


class DySurv72h(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_rnn: int = 64,
        layers_rnn: int = 1,
        latent_dim: int = 16,
        encoder_layers: list[int] | None = None,
        decoder_layers: list[int] | None = None,
        survival_layers: list[int] | None = None,
        dropout: float = 0.1,
        num_durations: int = 10,
    ):
        super().__init__()
        encoder_layers = encoder_layers or [128, 64]
        decoder_layers = decoder_layers or [128, 64]
        survival_layers = survival_layers or [64]
        self.input_dim = input_dim
        self.num_durations = num_durations
        self.encoder_rnn = nn.LSTM(
            input_dim,
            hidden_rnn,
            num_layers=layers_rnn,
            dropout=dropout if layers_rnn > 1 else 0.0,
            batch_first=True,
        )
        self.encoder_body = mlp(hidden_rnn, encoder_layers, encoder_layers[-1], dropout)
        self.mu = nn.Linear(encoder_layers[-1], latent_dim)
        self.logvar = nn.Linear(encoder_layers[-1], latent_dim)
        self.survival_head = mlp(latent_dim, survival_layers, num_durations, dropout)
        self.decoder = mlp(latent_dim, decoder_layers, 72 * input_dim, dropout)

    def encode(self, x):
        _, (hidden, _) = self.encoder_rnn(x)
        h_last = hidden[-1]
        encoded = self.encoder_body(h_last)
        return self.mu(encoded), self.logvar(encoded)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.survival_head(z)
        reconstruction = self.decoder(z).view(x.shape[0], x.shape[1], x.shape[2])
        return {"logits": logits, "reconstruction": reconstruction, "mu": mu, "logvar": logvar}

    def predict_logits(self, x):
        mu, logvar = self.encode(x)
        return self.survival_head(mu)


def kl_loss(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

