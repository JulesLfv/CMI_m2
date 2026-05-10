from typing import List

import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, window_size: int, hidden_sizes: List[int], output_dim=2, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim * window_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))


class RNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_size=128, num_layers=2, output_dim=2, dropout=0.2, rnn_type="LSTM"):
        super().__init__()
        klass = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_type]
        self.rnn = klass(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class TCNRegressor(nn.Module):
    def __init__(self, input_dim, channels=(64, 128), kernel_size=3, dropout=0.2, output_dim=2):
        super().__init__()
        layers = []
        c_in = input_dim
        for c_out in channels:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.BatchNorm1d(c_out),
                nn.Dropout(dropout),
            ]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(c_in, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        z = self.conv(x)
        z = self.pool(z).squeeze(-1)
        return self.head(z)
