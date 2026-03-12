import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.ip102_dataset import IP102Dataset
from utils.experiment_utils import TeeLogger, find_latest_experiment, load_experiment_config, resolve_checkpoint
from utils.train_utils import AverageMeter, load_checkpoint


DEFAULTS = {
    "experiment_root": "experiments",
    "task_name": "baseline",
    "metadata_csv": "data/ip102/metadata.csv",
    "images_dir": "data/ip102/images",
    "num_classes": 102,
    "batch_size": 32,
    "num_workers": 4,
    "image_size": 224,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--experiment_root", type=str, default=DEFAULTS["experiment_root"])
    parser.add_argument("--task_name", type=str, default=DEFAULTS["task_name"])
    parser.add_argument("--save_name", type=str, default="test_metrics.json")
    return parser.parse_args()



def main():
    args = parse_args()
    checkpoint_path = resolve_checkpoint(args.checkpoint, args.exp_dir, args.experiment_root, args.task_name, prefer="best")
    exp_dir = args.exp_dir or str(Path(checkpoint_path).resolve().parents[1])
    config = DEFAULTS.copy()
    if (Path(exp_dir) / "config.json").exists():
        config.update(load_experiment_config(exp_dir))

    logger = TeeLogger(Path(exp_dir) / "logs" / "evaluate.log", logger_name=f"eval_{config['task_name']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating checkpoint: {checkpoint_path}")
    logger.info(f"Using device: {device}")

    test_transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
    ])
    test_dataset = IP102Dataset(config["metadata_csv"], config["images_dir"], split="test", transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, config["num_classes"])
    load_checkpoint(checkpoint_path, model, device=device)
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_meter.update(loss.item(), labels.size(0))
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    metrics = {
        "test_loss": loss_meter.avg,
        "test_acc": 100.0 * correct / max(total, 1),
        "checkpoint": checkpoint_path,
    }
    out_path = Path(exp_dir) / "metrics" / args.save_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
