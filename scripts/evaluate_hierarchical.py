import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.ip102_hier_dataset import IP102HierDataset
from utils.experiment_utils import TeeLogger, load_experiment_config, resolve_checkpoint
from utils.train_utils import AverageMeter, load_checkpoint


DEFAULTS = {
    "experiment_root": "experiments",
    "task_name": "hierarchical_baseline",
    "metadata_csv": "data/ip102/metadata_hierarchical.csv",
    "images_dir": "data/ip102/images",
    "num_label_classes": 102,
    "batch_size": 32,
    "num_workers": 4,
    "image_size": 224,
}


class HierResNet(nn.Module):
    def __init__(self, num_label_classes, num_genus_classes, num_family_classes, num_order_classes):
        super().__init__()
        backbone = resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.label_head = nn.Linear(in_features, num_label_classes)
        self.genus_head = nn.Linear(in_features, num_genus_classes)
        self.family_head = nn.Linear(in_features, num_family_classes)
        self.order_head = nn.Linear(in_features, num_order_classes)

    def forward(self, x):
        feats = self.backbone(x)
        return {
            "label": self.label_head(feats),
            "genus": self.genus_head(feats),
            "family": self.family_head(feats),
            "order": self.order_head(feats),
        }



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--experiment_root", type=str, default=DEFAULTS["experiment_root"])
    parser.add_argument("--task_name", type=str, default=DEFAULTS["task_name"])
    parser.add_argument("--save_name", type=str, default="test_metrics.json")
    return parser.parse_args()



def taxonomy_sizes(metadata_csv):
    df = pd.read_csv(metadata_csv)
    return {
        "num_genus_classes": int(df["genus_id"].nunique()),
        "num_family_classes": int(df["family_id"].nunique()),
        "num_order_classes": int(df["order_id"].nunique()),
    }



def main():
    args = parse_args()
    checkpoint_path = resolve_checkpoint(args.checkpoint, args.exp_dir, args.experiment_root, args.task_name, prefer="best")
    exp_dir = args.exp_dir or str(Path(checkpoint_path).resolve().parents[1])

    config = DEFAULTS.copy()
    if (Path(exp_dir) / "config.json").exists():
        config.update(load_experiment_config(exp_dir))
    config.update(taxonomy_sizes(config["metadata_csv"]))

    logger = TeeLogger(Path(exp_dir) / "logs" / "evaluate.log", logger_name=f"eval_{config['task_name']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating checkpoint: {checkpoint_path}")
    logger.info(f"Using device: {device}")

    test_transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
    ])
    test_dataset = IP102HierDataset(config["metadata_csv"], config["images_dir"], split="test", transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    model = HierResNet(
        num_label_classes=config["num_label_classes"],
        num_genus_classes=config["num_genus_classes"],
        num_family_classes=config["num_family_classes"],
        num_order_classes=config["num_order_classes"],
    )
    load_checkpoint(checkpoint_path, model, device=device)
    model = model.to(device)
    model.eval()

    ce = nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    label_correct = genus_correct = family_correct = order_correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            genus_ids = batch["genus_id"].to(device, non_blocking=True)
            family_ids = batch["family_id"].to(device, non_blocking=True)
            order_ids = batch["order_id"].to(device, non_blocking=True)

            outputs = model(images)
            loss = ce(outputs["label"], labels)
            loss_meter.update(loss.item(), labels.size(0))

            label_correct += (outputs["label"].argmax(dim=1) == labels).sum().item()
            genus_correct += (outputs["genus"].argmax(dim=1) == genus_ids).sum().item()
            family_correct += (outputs["family"].argmax(dim=1) == family_ids).sum().item()
            order_correct += (outputs["order"].argmax(dim=1) == order_ids).sum().item()
            total += labels.size(0)

    metrics = {
        "test_loss_label_head": loss_meter.avg,
        "test_label_acc": 100.0 * label_correct / max(total, 1),
        "test_genus_acc": 100.0 * genus_correct / max(total, 1),
        "test_family_acc": 100.0 * family_correct / max(total, 1),
        "test_order_acc": 100.0 * order_correct / max(total, 1),
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
