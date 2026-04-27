from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np
import torch


@dataclass
class TransitionBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.storage: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.storage)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.storage.append((np.asarray(state, dtype=np.float32), int(action), float(reward), np.asarray(next_state, dtype=np.float32), bool(done)))

    def sample(self, batch_size: int, device: torch.device) -> TransitionBatch:
        indices = np.random.choice(len(self.storage), size=int(batch_size), replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.storage[index] for index in indices))
        return TransitionBatch(
            states=torch.as_tensor(np.stack(states), dtype=torch.float32, device=device),
            actions=torch.as_tensor(actions, dtype=torch.long, device=device),
            rewards=torch.as_tensor(rewards, dtype=torch.float32, device=device),
            next_states=torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=device),
            dones=torch.as_tensor(dones, dtype=torch.float32, device=device),
        )
