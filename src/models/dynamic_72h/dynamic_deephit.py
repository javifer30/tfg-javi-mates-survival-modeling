"""Dynamic-DeepHit adaptation for dynamic_72h.

Reference reviewed:
src/models_references/DynamicDeepHit/ddh/ddh_torch.py
src/models_references/DynamicDeepHit/ddh/losses.py

The implementation keeps the reference structure: recurrent embedding,
longitudinal prediction network, temporal attention, cause-specific network and
softmax PMF output. It is adapted to one risk and one prediction per patient at
the end of the first 72h.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def create_nn(input_dim, output_dim, layers=None, dropout=0.1, activation="ReLU", no_activation_last=False):
    layers = layers or [64]
    act = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "SeLU": nn.SELU}.get(activation, nn.ReLU)
    modules = []
    prev = input_dim
    for hidden in layers:
        modules.append(nn.Linear(prev, hidden))
        modules.append(act())
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        prev = hidden
    modules.append(nn.Linear(prev, output_dim))
    if not no_activation_last:
        modules.append(act())
    return nn.Sequential(*modules)


class DynamicDeepHit72h(nn.Module):
    def __init__(
        self,
        input_dim: int,
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
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.typ = typ
        if typ == "GRU":
            self.embedding = nn.GRU(input_dim, hidden_rnn, layers_rnn, batch_first=True)
        elif typ == "RNN":
            self.embedding = nn.RNN(input_dim, hidden_rnn, layers_rnn, nonlinearity="relu", batch_first=True)
        else:
            self.embedding = nn.LSTM(input_dim, hidden_rnn, layers_rnn, batch_first=True)
        self.longitudinal = create_nn(hidden_rnn, input_dim, no_activation_last=True, **long_param)
        self.attention = create_nn(input_dim + hidden_rnn, 1, no_activation_last=True, **att_param)
        self.attention_soft = nn.Softmax(dim=1)
        self.cause_specific = create_nn(input_dim + hidden_rnn, output_dim, no_activation_last=True, **cs_param)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        hidden_out = self.embedding(x)
        hidden = hidden_out[0] if isinstance(hidden_out, tuple) else hidden_out
        longitudinal_prediction = self.longitudinal(hidden)
        x_last = x[:, -1, :]
        concat = torch.cat([hidden, x_last.unsqueeze(1).repeat(1, x.shape[1], 1)], dim=2)
        attention = self.attention(concat).squeeze(-1)
        # all 72 timesteps are valid in the TFG tensor; attention summarizes
        # the previous context for the final prediction.
        attention = self.attention_soft(attention)
        hidden_attentive = torch.sum(attention.unsqueeze(2) * hidden, dim=1)
        risk_input = torch.cat([hidden_attentive, x_last], dim=1)
        pmf = self.soft(self.cause_specific(risk_input))
        return {"longitudinal_prediction": longitudinal_prediction, "pmf": pmf}
