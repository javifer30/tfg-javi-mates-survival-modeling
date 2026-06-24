"""Faithful DySurv adaptation for the landmark experiment.

Reference:
src/models_references/DySurv/Models/Results/DySurv.ipynb

Preserved structure: LSTM encoder, 3x/5x/3x encoder MLP, variational latent
state, discrete LogisticHazard head, and recurrent sequence decoder. The input
sequence contains temporal clinical variables plus repeated static covariates.
The outcome duration used by the reference decoder is deliberately removed to
prevent target leakage. The decoder reconstructs temporal clinical variables
only; masks and repeated static variables are never reconstruction targets.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _activation(name: str) -> nn.Module:
    if name.lower() == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


def _mlp(input_dim: int, layers: list[int], output_dim: int, dropout: float) -> nn.Sequential:
    modules: list[nn.Module] = []
    previous = input_dim
    for hidden in layers:
        modules.extend([nn.Linear(previous, int(hidden)), nn.ReLU()])
        if dropout > 0:
            modules.append(nn.Dropout(float(dropout)))
        previous = int(hidden)
    modules.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*modules)


class RecurrentTemporalDecoder(nn.Module):
    """Decode the latent state back into a temporal clinical trajectory."""

    def __init__(self, latent_dim: int, output_dim: int, seq_len: int, dropout: float):
        super().__init__()
        hidden_dim = 2 * latent_dim
        self.seq_len = int(seq_len)
        self.rnn = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.output = _mlp(
            hidden_dim,
            [3 * hidden_dim, 5 * hidden_dim, 3 * hidden_dim],
            output_dim,
            dropout,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        repeated = z.unsqueeze(1).expand(-1, self.seq_len, -1)
        decoded, _ = self.rnn(repeated)
        return self.output(decoded)


class DySurvFaithful72h(nn.Module):
    """Temporal DySurv model used for 24h/48h/72h landmark runs."""

    def __init__(
        self,
        input_dim: int,
        reconstruction_dim: int,
        seq_len: int = 72,
        rnn_hidden_dim: int | None = None,
        latent_dim: int = 20,
        encoder_mlp: list[int] | None = None,
        survival_mlp: list[int] | None = None,
        dropout: float = 0.1,
        num_durations: int = 10,
    ):
        super().__init__()
        encoder_mlp = encoder_mlp or [294, 490, 294]
        survival_mlp = survival_mlp or [294, 490, 294]
        hidden_dim = int(rnn_hidden_dim or max(98, input_dim))
        self.input_dim = int(input_dim)
        self.reconstruction_dim = int(reconstruction_dim)
        self.latent_dim = int(latent_dim)
        self.num_durations = int(num_durations)
        self.encoder_rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.encoder_body = _mlp(hidden_dim, encoder_mlp[:-1], encoder_mlp[-1], dropout)
        self.mu = nn.Linear(encoder_mlp[-1], latent_dim)
        self.logvar = nn.Linear(encoder_mlp[-1], latent_dim)
        self.survival_head = _mlp(latent_dim, survival_mlp, num_durations, dropout)
        self.decoder = RecurrentTemporalDecoder(latent_dim, reconstruction_dim, seq_len, dropout)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, (hidden, _) = self.encoder_rnn(x)
        encoded = self.encoder_body(hidden[-1])
        return self.mu(encoded), self.logvar(encoded)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return {
            "reconstruction": self.decoder(z),
            "logits": self.survival_head(z),
            "mu": mu,
            "logvar": logvar,
        }

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Use deterministic latent means for evaluation and prediction."""
        mu, _ = self.encode(x)
        return self.survival_head(mu)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))
