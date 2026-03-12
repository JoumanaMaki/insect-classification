import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


class HistoryTracker:
    def __init__(self) -> None:
        self.history: Dict[str, List[float]] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            self.history.setdefault(k, []).append(float(v))

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def get(self) -> Dict[str, List[float]]:
        return self.history


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[object],
    epoch: int,
    metrics: Dict[str, float],
    config: Dict,
    path: str,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None and hasattr(scheduler, "state_dict") else None,
        "metrics": metrics,
        "config": config,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(checkpoint, path)



def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, scheduler=None, device: str = "cpu") -> Dict:
    checkpoint = torch.load(path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint

    model.load_state_dict(checkpoint)
    return {"epoch": 0, "metrics": {}, "config": {}}



def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return 100.0 * (preds == targets).sum().item() / max(targets.size(0), 1)



def print_epoch_metrics(epoch: int, epochs: int, metrics: Dict[str, float], epoch_seconds: Optional[float] = None) -> str:
    parts = [f"Epoch {epoch}/{epochs}"]
    for k, v in metrics.items():
        parts.append(f"{k}={v:.4f}")
    if epoch_seconds is not None:
        parts.append(f"epoch_time_sec={epoch_seconds:.2f}")
    return " | ".join(parts)
