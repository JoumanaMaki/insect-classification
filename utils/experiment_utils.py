import argparse
import json
import logging
import os
import random
import shutil
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


DEFAULT_EXPERIMENT_ROOT = "experiments"


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class TeeLogger:
    """Create a logger that writes both to stdout and to a log file."""

    def __init__(self, log_file: Path, logger_name: str = "ip102") -> None:
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(f"{logger_name}_{self.log_file.stem}_{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        self.logger.handlers.clear()
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)



def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True



def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def get_torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator



def _to_plain_dict(config: Any) -> Dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, argparse.Namespace):
        return vars(config)
    if isinstance(config, dict):
        return deepcopy(config)
    raise TypeError(f"Unsupported config type: {type(config)}")



def load_json_config(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)



def merge_config(defaults: Dict[str, Any], config_file_data: Dict[str, Any], cli_args: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(defaults)
    merged.update(config_file_data)
    for key, value in cli_args.items():
        if value is not None:
            merged[key] = value
    return merged



def create_experiment_dir(base_dir: str, task_name: str, run_name: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Path]:
    timestamp = now_timestamp()
    safe_task = task_name.lower().replace(" ", "_")
    tail_parts = [timestamp]
    if run_name:
        tail_parts.append(str(run_name))
    if seed is not None:
        tail_parts.append(f"seed{seed}")
    exp_name = "_".join(tail_parts)

    root = Path(base_dir) / safe_task / exp_name
    paths = {
        "exp_dir": root,
        "checkpoints": root / "checkpoints",
        "logs": root / "logs",
        "metrics": root / "metrics",
        "plots": root / "plots",
        "artifacts": root / "artifacts",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths



def save_config(config: Any, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_plain_dict(config), f, indent=2, sort_keys=True)



def copy_config_file(config_path: Optional[str], destination_dir: str) -> Optional[str]:
    if not config_path:
        return None
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / Path(config_path).name
    shutil.copy2(config_path, destination)
    return str(destination)



def load_experiment_config(exp_dir: str) -> Dict[str, Any]:
    config_path = Path(exp_dir) / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)



def find_latest_experiment(base_dir: str, task_name: str) -> Optional[str]:
    task_dir = Path(base_dir) / task_name.lower().replace(" ", "_")
    if not task_dir.exists():
        return None
    candidates = [p for p in task_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    latest = sorted(candidates, key=lambda p: p.name)[-1]
    return str(latest)



def resolve_checkpoint(
    checkpoint: Optional[str] = None,
    exp_dir: Optional[str] = None,
    base_dir: str = DEFAULT_EXPERIMENT_ROOT,
    task_name: Optional[str] = None,
    prefer: str = "best",
) -> str:
    if checkpoint:
        return checkpoint

    if exp_dir:
        ckpt = Path(exp_dir) / "checkpoints" / f"{prefer}.pth"
        if ckpt.exists():
            return str(ckpt)
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if task_name:
        latest = find_latest_experiment(base_dir, task_name)
        if latest is None:
            raise FileNotFoundError(f"No experiments found under {base_dir}/{task_name}")
        ckpt = Path(latest) / "checkpoints" / f"{prefer}.pth"
        if ckpt.exists():
            return str(ckpt)
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    raise ValueError("Provide checkpoint, exp_dir, or task_name to resolve a checkpoint.")
