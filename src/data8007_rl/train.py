import csv
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import numpy as np
import pandas as pd

from data8007_rl.agent import DQNAgent
from data8007_rl.config import method_label
from data8007_rl.evaluate import evaluate_policy
from data8007_rl.replay_buffer import ReplayBuffer
from data8007_rl.utils import ensure_dir, get_device, moving_average, save_json, set_global_seed


def epsilon_at_step(global_step: int, config: Dict) -> float:
    exploration = config["exploration"]
    start = float(exploration["epsilon_start"])
    end = float(exploration["epsilon_end"])
    decay_steps = max(1, int(exploration["epsilon_decay_steps"]))
    fraction = min(1.0, float(global_step) / float(decay_steps))
    return start + fraction * (end - start)


def train_single_seed(config: Dict, seed: int) -> Dict:
    set_global_seed(int(seed))
    device = get_device(str(config.get("device", "auto")))
    method = str(config["method"])
    env_id = str(config["env_id"])
    env = gym.make(env_id)
    state, _ = env.reset(seed=int(seed))
    env.action_space.seed(int(seed))
    state_dim = int(np.asarray(state).shape[0])
    action_dim = int(env.action_space.n)
    hidden_sizes = config["network"]["hidden_sizes"]
    agent = DQNAgent(state_dim, action_dim, hidden_sizes, config, device, double_dqn=(method == "double_dqn"))
    replay_buffer = ReplayBuffer(int(config["optimization"]["replay_capacity"]), state_dim)

    log_dir = ensure_dir(config["paths"]["log_dir"])
    result_dir = ensure_dir(config["paths"]["result_dir"])
    checkpoint_dir = ensure_dir(config["paths"]["checkpoint_dir"])
    run_name = f"{method}_seed{seed}"
    train_log_path = log_dir / f"train_{run_name}.csv"
    eval_log_path = log_dir / f"eval_{run_name}.csv"
    best_checkpoint_path = checkpoint_dir / f"{run_name}_best.pt"
    final_checkpoint_path = checkpoint_dir / f"{run_name}_final.pt"

    episode_returns: List[float] = []
    global_step = 0
    best_eval_return = float("-inf")
    best_eval_episode = 0
    solved_episode = None
    max_episodes = int(config["max_episodes"])
    max_steps = int(config["max_steps_per_episode"])
    learning_starts = int(config["optimization"]["learning_starts"])
    train_frequency = int(config["optimization"]["train_frequency"])
    target_update_interval = int(config["optimization"]["target_update_interval"])
    eval_interval = int(config["evaluation"]["eval_interval"])
    eval_episodes = int(config["evaluation"]["eval_episodes"])
    solved_return = float(config["solved_return"])

    with train_log_path.open("w", newline="", encoding="utf-8") as train_handle, eval_log_path.open("w", newline="", encoding="utf-8") as eval_handle:
        train_fields = ["method", "seed", "episode", "global_step", "return", "length", "epsilon", "loss_mean", "moving_avg_50", "device"]
        eval_fields = ["method", "seed", "episode", "global_step", "mean_return", "std_return", "mean_length", "device"]
        train_writer = csv.DictWriter(train_handle, fieldnames=train_fields)
        eval_writer = csv.DictWriter(eval_handle, fieldnames=eval_fields)
        train_writer.writeheader()
        eval_writer.writeheader()

        for episode in range(1, max_episodes + 1):
            if episode == 1:
                state, _ = env.reset(seed=int(seed))
            else:
                state, _ = env.reset()
            episode_return = 0.0
            episode_losses: List[float] = []
            for step_in_episode in range(max_steps):
                epsilon = epsilon_at_step(global_step, config)
                action = agent.select_action(np.asarray(state, dtype=np.float32), epsilon=epsilon)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_return += float(reward)
                global_step += 1
                if global_step >= learning_starts and global_step % train_frequency == 0:
                    loss = agent.update(replay_buffer)
                    if loss > 0.0:
                        episode_losses.append(loss)
                if global_step % target_update_interval == 0:
                    agent.sync_target()
                if done:
                    break

            episode_returns.append(episode_return)
            moving_avg = moving_average(episode_returns, window=50)[-1]
            if solved_episode is None and moving_avg >= solved_return:
                solved_episode = episode
            train_writer.writerow({
                "method": method,
                "seed": seed,
                "episode": episode,
                "global_step": global_step,
                "return": episode_return,
                "length": step_in_episode + 1,
                "epsilon": epsilon_at_step(global_step, config),
                "loss_mean": float(np.mean(episode_losses)) if episode_losses else 0.0,
                "moving_avg_50": moving_avg,
                "device": str(device),
            })
            train_handle.flush()

            if episode % eval_interval == 0 or episode == max_episodes:
                metrics = evaluate_policy(env_id, agent, seed=int(seed) + episode, episodes=eval_episodes, max_steps=max_steps)
                eval_return = metrics["mean_return"]
                eval_writer.writerow({
                    "method": method,
                    "seed": seed,
                    "episode": episode,
                    "global_step": global_step,
                    "mean_return": eval_return,
                    "std_return": metrics["std_return"],
                    "mean_length": metrics["mean_length"],
                    "device": str(device),
                })
                eval_handle.flush()
                print(f"{method_label(method)} seed={seed} episode={episode} eval_return={eval_return:.1f} moving_avg_50={moving_avg:.1f} device={device}", flush=True)
                if eval_return > best_eval_return:
                    best_eval_return = eval_return
                    best_eval_episode = episode
                    agent.save(str(best_checkpoint_path), {"config": config, "seed": seed, "method": method, "episode": episode, "state_dim": state_dim, "action_dim": action_dim, "eval_return": best_eval_return})

    env.close()
    agent.save(str(final_checkpoint_path), {"config": config, "seed": seed, "method": method, "episode": max_episodes, "state_dim": state_dim, "action_dim": action_dim, "eval_return": best_eval_return})
    summary = {
        "method": method,
        "seed": int(seed),
        "device": str(device),
        "episodes": max_episodes,
        "global_step": int(global_step),
        "final_return": float(episode_returns[-1]),
        "final_moving_avg_50": float(moving_average(episode_returns, window=50)[-1]),
        "best_eval_return": float(best_eval_return),
        "best_eval_episode": int(best_eval_episode),
        "solved_episode": int(solved_episode) if solved_episode is not None else None,
        "train_log": str(train_log_path),
        "eval_log": str(eval_log_path),
        "best_checkpoint": str(best_checkpoint_path),
        "final_checkpoint": str(final_checkpoint_path),
    }
    save_json(str(result_dir / f"summary_{run_name}.json"), summary)
    return summary


def train_experiment(config: Dict) -> List[Dict]:
    summaries = []
    for seed in config["seeds"]:
        summaries.append(train_single_seed(config, int(seed)))
    result_dir = ensure_dir(config["paths"]["result_dir"])
    method = str(config["method"])
    pd.DataFrame(summaries).to_csv(result_dir / f"summary_{method}.csv", index=False)
    return summaries
