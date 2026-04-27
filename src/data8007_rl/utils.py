import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch


def ensure_dir(path: str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preferred == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(preferred)


def moving_average(values: Iterable[float], window: int) -> List[float]:
    output = []
    buffer = []
    for value in values:
        buffer.append(float(value))
        if len(buffer) > window:
            buffer.pop(0)
        output.append(float(np.mean(buffer)))
    return output


def save_json(path: str, payload: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
