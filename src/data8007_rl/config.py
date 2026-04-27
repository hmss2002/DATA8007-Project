from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["config_path"] = str(config_path)
    return config


def method_label(method: str) -> str:
    labels = {"dqn": "DQN", "double_dqn": "Double DQN"}
    return labels.get(method, method)
