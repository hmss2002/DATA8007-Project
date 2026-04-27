# DATA8007 Project: DQN vs Double DQN on CartPole-v1

This repository is a reproducible course project for DATA8007 Foundations of Sequential Decision-Making [2A, 2025].

## Project plan

### Topic

The project studies value-based deep reinforcement learning on the Gymnasium CartPole-v1 control task. The baseline method is Deep Q-Network (DQN) with experience replay and a target network. The proposed improvement is Double DQN, which decouples action selection from action evaluation in the temporal-difference target.

### Motivation

DQN is a natural extension of tabular Q-learning to problems with continuous state spaces. It is simple enough to explain in a short course report, but it still includes central RL ideas: bootstrapping, off-policy learning, exploration, replay buffers, and target networks. CartPole-v1 is intentionally small, so experiments can run quickly across multiple random seeds without large compute.

### Main research questions

1. How reliably does vanilla DQN solve CartPole-v1 under fixed hyperparameters?
2. Does Double DQN improve stability or final return on this problem?
3. Are observed differences meaningful for a simple benchmark, or mostly within seed variance?
4. What are the strengths, limitations, and possible extensions of value-based deep RL in this setting?

### Expected outcome before experiments

Both methods should eventually reach high returns because CartPole-v1 is low-dimensional and has discrete actions. Double DQN may reduce overestimation bias and produce slightly more stable learning, but the advantage may be modest because the environment is simple and the action space has only two actions.

### Experimental design

- Environment: Gymnasium CartPole-v1
- Methods: DQN and Double DQN
- Framework: PyTorch
- Hardware: automatically uses CUDA when available, otherwise CPU
- Seeds: multiple random seeds, configured in YAML
- Metrics: episodic return, 50-episode moving average, evaluation return, and solved episode
- Outputs: CSV logs, JSON summaries, checkpoints, and comparison figures

### Reproducibility policy

Experiment logs and figures are saved under logs/, results/, and checkpoints/. Results should not be edited by hand. If experiments are rerun, the outputs should be rerun with the scripts.

## Repository layout

```text
configs/                 YAML experiment configs
src/data8007_rl/          Modular PyTorch RL implementation
scripts/                 Training, evaluation, plotting, and full pipeline scripts
logs/                    Per-episode and evaluation logs
results/                 Summaries and comparison plots
checkpoints/             Trained model weights
report/                  LaTeX project report
presentation/            20-minute presentation outline
```

## Quick start

The project provides both a conda-style environment.yml and a pip requirements.txt. On systems without conda, use the local virtual environment script.

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

Run the full experiment pipeline:

```bash
bash scripts/run_all.sh
```

Or run individual steps:

```bash
python scripts/train.py --config configs/dqn_cartpole.yaml
python scripts/train.py --config configs/double_dqn_cartpole.yaml
python scripts/plot_results.py --log-dir logs --output-dir results/figures
python scripts/evaluate.py --config configs/dqn_cartpole.yaml --checkpoint checkpoints/dqn_seed0_best.pt
```

## Experiment results

These numbers come from `bash scripts/run_all.sh` on the SSH GPU node with `torch 2.8.0+cu128`, `gymnasium 0.29.1`, and `NVIDIA L40S`. Results are saved in `results/summary.csv` and should be rerun through scripts, not edited by hand.

| Method | Seeds | Final return mean +/- std | Final 50-episode MA mean +/- std | Best eval return mean +/- std |
| --- | ---: | ---: | ---: | ---: |
| DQN | 3 | 90.33 +/- 37.45 | 38.37 +/- 5.46 | 166.60 +/- 48.91 |
| Double DQN | 3 | 160.33 +/- 57.84 | 60.09 +/- 17.75 | 250.93 +/- 74.79 |

No seed reached the configured solved threshold of 475 within 160 training episodes. In this short run, Double DQN produced higher aggregate final and best-evaluation returns, but the variance is large and the comparison should be discussed as a small-budget experiment rather than a definitive benchmark result.

Experiment outputs:

- `logs/train_*.csv` and `logs/eval_*.csv`
- `results/summary.csv`, method-level CSVs, and seed-level JSON summaries
- `results/figures/learning_curves.png`
- `results/figures/evaluation_curves.png`
- `results/evaluation_*_best.json` from fixed-seed checkpoint evaluation

## Current status

The project contains the planned modular code, configs, clean environment files, experiment logs/results/figures, a LaTeX report, presentation outline, and reproducibility scripts. Binary model checkpoints are saved locally under `checkpoints/` but ignored by git to keep the repository lightweight.
