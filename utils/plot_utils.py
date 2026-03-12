import os
import matplotlib.pyplot as plt


def plot_history(history: dict, output_dir: str, prefix: str = ""):
    os.makedirs(output_dir, exist_ok=True)

    for metric_name, values in history.items():
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(values) + 1), values)
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.title(metric_name)
        plt.grid(True)
        filename = f"{prefix}_{metric_name}.png" if prefix else f"{metric_name}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()