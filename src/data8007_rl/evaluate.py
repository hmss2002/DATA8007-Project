from typing import Dict, List

import gymnasium as gym
import numpy as np

from data8007_rl.agent import DQNAgent


def evaluate_policy(env_id: str, agent: DQNAgent, seed: int, episodes: int, max_steps: int) -> Dict[str, float]:
    env = gym.make(env_id)
    returns: List[float] = []
    lengths: List[int] = []
    for episode in range(int(episodes)):
        state, _ = env.reset(seed=int(seed) + 10000 + episode)
        episode_return = 0.0
        episode_length = 0
        for _ in range(int(max_steps)):
            action = agent.select_action(np.asarray(state, dtype=np.float32), epsilon=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            episode_length += 1
            state = next_state
            if terminated or truncated:
                break
        returns.append(episode_return)
        lengths.append(episode_length)
    env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "episodes": int(episodes),
    }
