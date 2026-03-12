import argparse
import json
import os
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.ip102_hier_dataset import IP102HierDataset
from utils.experiment_utils import (
    TeeLogger,
    copy_config_file,
    create_experiment_dir,
    get_torch_generator,
    load_json_config,
    merge_config,
    save_config,
    seed_worker,
    set_seed,
)
from utils.plot_utils import plot_history
from utils.train_utils import HistoryTracker, AverageMeter, print_epoch_metrics, save_checkpoint


DEFAULTS = {
    "experiment_root": "experiments",
    "task_name": "hierarchical_baseline",
    "run_name": "resnet18_multihead",
    "metadata_csv": "data/ip102/metadata_hierarchical.csv",
    "images_dir": "data/ip102/images",
    "num_label_classes": 102,
    "batch_size": 32,
    "num_workers": 4,
    "epochs": 30,
    "lr": 1e-4,
    "weight_decay": 0.0,
    "seed": 42,
    "deterministic": False,
    "image_size": 224,
    "resize_size": 256,
    "pretrained": True,
    "label_loss_weight": 1.0,
    "genus_loss_weight": 1.0,
    "family_loss_weight": 1.0,
    "order_loss_weight": 1.0,
    "save_every": 1,
}


class HierResNet(nn.Module):
    def __init__(self, num_label_classes, num_genus_classes, num_family_classes, num_order_classes, pretrained=True):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
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
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--experiment_root", type=str, default=None)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--metadata_csv", type=str, default=None)
    parser.add_argument("--images_dir", type=str, default=None)
    parser.add_argument("--num_label_classes", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--resize_size", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--label_loss_weight", type=float, default=None)
    parser.add_argument("--genus_loss_weight", type=float, default=None)
    parser.add_argument("--family_loss_weight", type=float, default=None)
    parser.add_argument("--order_loss_weight", type=float, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=None)
    return parser.parse_args()



def compute_taxonomy_sizes(metadata_csv):
    df = pd.read_csv(metadata_csv)
    return {
        "num_genus_classes": int(df["genus_id"].nunique()),
        "num_family_classes": int(df["family_id"].nunique()),
        "num_order_classes": int(df["order_id"].nunique()),
    }



def evaluate(model, loader, ce, device, cfg):
    model.eval()
    total_loss_meter = AverageMeter()
    label_correct = genus_correct = family_correct = order_correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            genus_ids = batch["genus_id"].to(device, non_blocking=True)
            family_ids = batch["family_id"].to(device, non_blocking=True)
            order_ids = batch["order_id"].to(device, non_blocking=True)

            outputs = model(images)
            loss = (
                cfg["label_loss_weight"] * ce(outputs["label"], labels)
                + cfg["genus_loss_weight"] * ce(outputs["genus"], genus_ids)
                + cfg["family_loss_weight"] * ce(outputs["family"], family_ids)
                + cfg["order_loss_weight"] * ce(outputs["order"], order_ids)
            )
            total_loss_meter.update(loss.item(), labels.size(0))

            label_correct += (outputs["label"].argmax(dim=1) == labels).sum().item()
            genus_correct += (outputs["genus"].argmax(dim=1) == genus_ids).sum().item()
            family_correct += (outputs["family"].argmax(dim=1) == family_ids).sum().item()
            order_correct += (outputs["order"].argmax(dim=1) == order_ids).sum().item()
            total += labels.size(0)

    return {
        "val_loss": total_loss_meter.avg,
        "val_label_acc": 100.0 * label_correct / max(total, 1),
        "val_genus_acc": 100.0 * genus_correct / max(total, 1),
        "val_family_acc": 100.0 * family_correct / max(total, 1),
        "val_order_acc": 100.0 * order_correct / max(total, 1),
    }



def main():
    args = parse_args()
    cfg = merge_config(DEFAULTS, load_json_config(args.config), vars(args))
    cfg.update(compute_taxonomy_sizes(cfg["metadata_csv"]))

    set_seed(cfg["seed"], deterministic=cfg["deterministic"])
    paths = create_experiment_dir(cfg["experiment_root"], cfg["task_name"], cfg["run_name"], cfg["seed"])
    logger = TeeLogger(paths["logs"] / "train.log", logger_name=cfg["task_name"])

    cfg["exp_dir"] = str(paths["exp_dir"])
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    save_config(cfg, paths["exp_dir"] / "config.json")
    copy_config_file(args.config, paths["artifacts"])

    device = torch.device(cfg["device"])
    logger.info(f"Experiment directory: {paths['exp_dir']}")
    logger.info(f"Using device: {device}")
    logger.info(json.dumps(cfg, indent=2))

    train_transform = transforms.Compose([
        transforms.Resize((cfg["resize_size"], cfg["resize_size"])),
        transforms.RandomCrop(cfg["image_size"]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
    ])

    train_dataset = IP102HierDataset(cfg["metadata_csv"], cfg["images_dir"], split="train", transform=train_transform)
    val_dataset = IP102HierDataset(cfg["metadata_csv"], cfg["images_dir"], split="val", transform=val_transform)
    logger.info(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=get_torch_generator(cfg["seed"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    model = HierResNet(
        num_label_classes=cfg["num_label_classes"],
        num_genus_classes=cfg["num_genus_classes"],
        num_family_classes=cfg["num_family_classes"],
        num_order_classes=cfg["num_order_classes"],
        pretrained=cfg["pretrained"],
    ).to(device)

    ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    tracker = HistoryTracker()
    best_val_label_acc = -1.0

    for epoch in range(1, cfg["epochs"] + 1):
        start = time.time()
        model.train()
        total_loss_meter = AverageMeter()
        label_correct = genus_correct = family_correct = order_correct = 0
        total = 0

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            genus_ids = batch["genus_id"].to(device, non_blocking=True)
            family_ids = batch["family_id"].to(device, non_blocking=True)
            order_ids = batch["order_id"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = (
                cfg["label_loss_weight"] * ce(outputs["label"], labels)
                + cfg["genus_loss_weight"] * ce(outputs["genus"], genus_ids)
                + cfg["family_loss_weight"] * ce(outputs["family"], family_ids)
                + cfg["order_loss_weight"] * ce(outputs["order"], order_ids)
            )
            loss.backward()
            optimizer.step()

            total_loss_meter.update(loss.item(), labels.size(0))
            label_correct += (outputs["label"].argmax(dim=1) == labels).sum().item()
            genus_correct += (outputs["genus"].argmax(dim=1) == genus_ids).sum().item()
            family_correct += (outputs["family"].argmax(dim=1) == family_ids).sum().item()
            order_correct += (outputs["order"].argmax(dim=1) == order_ids).sum().item()
            total += labels.size(0)

        scheduler.step()
        val_metrics = evaluate(model, val_loader, ce, device, cfg)
        metrics = {
            "train_loss": total_loss_meter.avg,
            "train_label_acc": 100.0 * label_correct / max(total, 1),
            "train_genus_acc": 100.0 * genus_correct / max(total, 1),
            "train_family_acc": 100.0 * family_correct / max(total, 1),
            "train_order_acc": 100.0 * order_correct / max(total, 1),
            **val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        tracker.update(metrics)
        epoch_seconds = time.time() - start
        logger.info(print_epoch_metrics(epoch, cfg["epochs"], metrics, epoch_seconds=epoch_seconds))

        save_checkpoint(model, optimizer, scheduler, epoch, metrics, cfg, str(paths["checkpoints"] / "last.pth"))
        if epoch % cfg["save_every"] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, metrics, cfg, str(paths["checkpoints"] / f"epoch_{epoch:03d}.pth"))
        if metrics["val_label_acc"] > best_val_label_acc:
            best_val_label_acc = metrics["val_label_acc"]
            save_checkpoint(model, optimizer, scheduler, epoch, metrics, cfg, str(paths["checkpoints"] / "best.pth"))
            logger.info(f"New best checkpoint at epoch {epoch} with val_label_acc={best_val_label_acc:.4f}")

    history_path = paths["metrics"] / "history.json"
    tracker.save_json(str(history_path))
    plot_history(tracker.get(), str(paths["plots"]), prefix=cfg["task_name"])
    logger.info(f"Training finished. Best val_label_acc={best_val_label_acc:.4f}")
    logger.info(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()
