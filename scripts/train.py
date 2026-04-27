import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data8007_rl.config import load_config
from data8007_rl.train import train_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN style agents for the DATA8007 project.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()
    config = load_config(args.config)
    train_experiment(config)


if __name__ == "__main__":
    main()
