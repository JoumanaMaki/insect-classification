import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.ip102_dataset import IP102Dataset
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
from utils.train_utils import HistoryTracker, AverageMeter, accuracy_from_logits, print_epoch_metrics, save_checkpoint


DEFAULTS = {
    "experiment_root": "experiments",
    "task_name": "baseline",
    "run_name": "resnet18",
    "metadata_csv": "data/ip102/metadata.csv",
    "images_dir": "data/ip102/images",
    "num_classes": 102,
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
    "save_every": 1,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--experiment_root", type=str, default=None)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--metadata_csv", type=str, default=None)
    parser.add_argument("--images_dir", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--resize_size", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=None)
    return parser.parse_args()


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss_meter.update(loss.item(), labels.size(0))
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return {
        "val_loss": loss_meter.avg,
        "val_acc": 100.0 * correct / max(total, 1),
    }


def main():
    args = parse_args()
    cfg = merge_config(DEFAULTS, load_json_config(args.config), vars(args))

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

    train_dataset = IP102Dataset(cfg["metadata_csv"], cfg["images_dir"], split="train", transform=train_transform)
    val_dataset = IP102Dataset(cfg["metadata_csv"], cfg["images_dir"], split="val", transform=val_transform)
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

    weights = ResNet18_Weights.DEFAULT if cfg["pretrained"] else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, cfg["num_classes"])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    tracker = HistoryTracker()
    best_val_acc = -1.0

    for epoch in range(1, cfg["epochs"] + 1):
        start = time.time()
        model.train()
        train_loss_meter = AverageMeter()
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_meter.update(loss.item(), labels.size(0))
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        val_metrics = evaluate(model, val_loader, criterion, device)
        metrics = {
            "train_loss": train_loss_meter.avg,
            "train_acc": 100.0 * train_correct / max(train_total, 1),
            "val_loss": val_metrics["val_loss"],
            "val_acc": val_metrics["val_acc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        tracker.update(metrics)
        epoch_seconds = time.time() - start
        logger.info(print_epoch_metrics(epoch, cfg["epochs"], metrics, epoch_seconds=epoch_seconds))

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=metrics,
            config=cfg,
            path=str(paths["checkpoints"] / "last.pth"),
        )
        if epoch % cfg["save_every"] == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                config=cfg,
                path=str(paths["checkpoints"] / f"epoch_{epoch:03d}.pth"),
            )
        if metrics["val_acc"] > best_val_acc:
            best_val_acc = metrics["val_acc"]
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                config=cfg,
                path=str(paths["checkpoints"] / "best.pth"),
            )
            logger.info(f"New best checkpoint at epoch {epoch} with val_acc={best_val_acc:.4f}")

    history_path = paths["metrics"] / "history.json"
    tracker.save_json(str(history_path))
    plot_history(tracker.get(), str(paths["plots"]), prefix=cfg["task_name"])
    logger.info(f"Training finished. Best val_acc={best_val_acc:.4f}")
    logger.info(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()
