import os
from datetime import datetime


def create_experiment_dir(base_dir: str, name: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{name}_{timestamp}")

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)

    return exp_dir