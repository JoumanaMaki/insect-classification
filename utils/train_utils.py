import os
import json
from typing import Dict, List

import torch


class HistoryTracker:
    def __init__(self):
        self.history = {}

    def update(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(float(v))

    def save_json(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def get(self) -> Dict[str, List[float]]:
        return self.history


def save_checkpoint(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint: {path}")


def print_epoch_metrics(epoch: int, epochs: int, metrics: Dict[str, float]):
    parts = [f"Epoch {epoch}/{epochs}"]
    for k, v in metrics.items():
        parts.append(f"{k}: {v:.4f}")
    print(" | ".join(parts))