from typing import Iterable

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Iterable[int]):
        super().__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            input_dim = int(hidden_dim)
        layers.append(nn.Linear(input_dim, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.model(states)
