import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data8007_rl.agent import DQNAgent
from data8007_rl.config import load_config
from data8007_rl.evaluate import evaluate_policy
from data8007_rl.utils import get_device, save_json, set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved DQN checkpoint.")
    parser.add_argument("--config", required=True, help="Config used to define the agent.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of greedy evaluation episodes.")
    parser.add_argument("--seed", type=int, default=12345, help="Evaluation seed.")
    args = parser.parse_args()
    config = load_config(args.config)
    set_global_seed(args.seed)
    device = get_device(str(config.get("device", "auto")))
    env = gym.make(config["env_id"])
    state, _ = env.reset(seed=args.seed)
    state_dim = int(np.asarray(state).shape[0])
    action_dim = int(env.action_space.n)
    env.close()
    agent = DQNAgent(state_dim, action_dim, config["network"]["hidden_sizes"], config, device, double_dqn=(config["method"] == "double_dqn"))
    checkpoint = agent.load(args.checkpoint)
    metrics = evaluate_policy(config["env_id"], agent, seed=args.seed, episodes=args.episodes, max_steps=int(config["max_steps_per_episode"]))
    metrics["checkpoint"] = args.checkpoint
    metrics["method"] = str(checkpoint.get("method", config["method"]))
    metrics["device"] = str(device)
    output = Path(config["paths"]["result_dir"]) / f"evaluation_{Path(args.checkpoint).stem}.json"
    save_json(str(output), metrics)
    print(metrics)


if __name__ == "__main__":
    main()
