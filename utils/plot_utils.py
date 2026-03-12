import os
from typing import Dict, List

import matplotlib.pyplot as plt



def _save_plot(x: List[int], y: List[float], title: str, ylabel: str, path: str) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()



def plot_history(history: Dict[str, List[float]], output_dir: str, prefix: str = "") -> None:
    os.makedirs(output_dir, exist_ok=True)
    if not history:
        return

    epochs = list(range(1, len(next(iter(history.values()))) + 1))

    for metric_name, values in history.items():
        filename = f"{prefix}_{metric_name}.png" if prefix else f"{metric_name}.png"
        _save_plot(epochs, values, metric_name.replace("_", " ").title(), metric_name, os.path.join(output_dir, filename))

    loss_keys = [k for k in history if "loss" in k]
    if loss_keys:
        plt.figure(figsize=(7, 5))
        for k in loss_keys:
            plt.plot(epochs, history[k], marker="o", label=k)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curves")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        filename = f"{prefix}_loss_curves.png" if prefix else "loss_curves.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close()

    acc_keys = [k for k in history if "acc" in k]
    if acc_keys:
        plt.figure(figsize=(7, 5))
        for k in acc_keys:
            plt.plot(epochs, history[k], marker="o", label=k)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy Curves")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        filename = f"{prefix}_accuracy_curves.png" if prefix else "accuracy_curves.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close()
