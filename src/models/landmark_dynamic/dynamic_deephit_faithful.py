"""Dynamic-DeepHit adaptation for the faithful 72-hour dataset.

Reference implementations:
- src/models_references/DynamicDeepHit/ddh/ddh_torch.py
- src/models_references/DynamicDeepHit/ddh/losses.py

The recurrent embedding, longitudinal prediction, temporal attention,
cause-specific network and PMF output are preserved. For the TFG landmark
experiment, static covariates may be repeated as encoder inputs, but the
longitudinal auxiliary task predicts only temporal clinical variables. This
prevents repeated static values from dominating the auxiliary MSE.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.landmark_dynamic.common import create_nn


class DynamicDeepHitFaithful72h(nn.Module):
    def __init__(
        self,
        input_dim: int,
        temporal_dim: int,
        output_dim: int,
        layers_rnn: int = 1,
        hidden_rnn: int = 64,
        typ: str = "LSTM",
        long_param: dict | None = None,
        att_param: dict | None = None,
        cs_param: dict | None = None,
    ):
        super().__init__()
        long_param = long_param or {"layers": [64], "dropout": 0.1, "activation": "ReLU"}
        att_param = att_param or {"layers": [64], "dropout": 0.1, "activation": "ReLU"}
        cs_param = cs_param or {"layers": [64], "dropout": 0.1, "activation": "ReLU"}
        self.input_dim = int(input_dim)
        self.temporal_dim = int(temporal_dim)
        self.output_dim = int(output_dim)
        self.typ = typ
        if typ == "GRU":
            self.embedding = nn.GRU(input_dim, hidden_rnn, layers_rnn, bias=False, batch_first=True)
        elif typ == "RNN":
            self.embedding = nn.RNN(input_dim, hidden_rnn, layers_rnn, bias=False, batch_first=True, nonlinearity="relu")
        else:
            self.embedding = nn.LSTM(input_dim, hidden_rnn, layers_rnn, bias=False, batch_first=True)
        self.longitudinal = create_nn(hidden_rnn, temporal_dim, no_activation_last=True, **long_param)
        self.attention = create_nn(input_dim + hidden_rnn, 1, no_activation_last=True, **att_param)
        self.attention_soft = nn.Softmax(dim=1)
        self.cause_specific = create_nn(input_dim + hidden_rnn, output_dim, no_activation_last=True, **cs_param)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden_output = self.embedding(x)
        hidden = hidden_output[0] if isinstance(hidden_output, tuple) else hidden_output
        longitudinal_prediction = self.longitudinal(hidden)
        x_last = x[:, -1, :]
        repeated_last = x_last.unsqueeze(1).expand(-1, x.shape[1], -1)
        attention_logits = self.attention(torch.cat([hidden, repeated_last], dim=2)).squeeze(-1)
        attention = self.attention_soft(attention_logits)
        hidden_attentive = torch.sum(attention.unsqueeze(2) * hidden, dim=1)
        risk_input = torch.cat([hidden_attentive, x_last], dim=1)
        pmf = self.softmax(self.cause_specific(risk_input))
        return {
            "longitudinal_prediction": longitudinal_prediction,
            "attention": attention,
            "pmf": pmf,
        }

    def predict_pmf(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["pmf"]
