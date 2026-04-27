from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from torch.nn import functional as F

from data8007_rl.networks import QNetwork
from data8007_rl.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Iterable[int], config: Dict, device: torch.device, double_dqn: bool):
        self.action_dim = int(action_dim)
        self.device = device
        self.double_dqn = bool(double_dqn)
        self.gamma = float(config["optimization"]["gamma"])
        self.batch_size = int(config["optimization"]["batch_size"])
        self.gradient_clip_norm = float(config["optimization"]["gradient_clip_norm"])
        self.online_network = QNetwork(state_dim, action_dim, hidden_sizes).to(device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_sizes).to(device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=float(config["optimization"]["learning_rate"]))

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if np.random.random() < float(epsilon):
            return int(np.random.randint(self.action_dim))
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def update(self, replay_buffer: ReplayBuffer) -> float:
        if len(replay_buffer) < self.batch_size:
            return 0.0
        batch = replay_buffer.sample(self.batch_size, self.device)
        q_values = self.online_network(batch.states).gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.online_network(batch.next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_network(batch.next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q_values = self.target_network(batch.next_states).max(dim=1).values
            targets = batch.rewards + self.gamma * (1.0 - batch.dones) * next_q_values
        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), self.gradient_clip_norm)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target_network.load_state_dict(self.online_network.state_dict())

    def save(self, path: str, extra: Dict) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(extra)
        payload["model_state_dict"] = self.online_network.state_dict()
        payload["target_state_dict"] = self.target_network.state_dict()
        torch.save(payload, output_path)

    def load(self, path: str) -> Dict:
        checkpoint = torch.load(path, map_location=self.device)
        self.online_network.load_state_dict(checkpoint["model_state_dict"])
        target_state = checkpoint.get("target_state_dict", checkpoint["model_state_dict"])
        self.target_network.load_state_dict(target_state)
        self.target_network.eval()
        return checkpoint
