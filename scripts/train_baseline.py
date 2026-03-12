import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.ip102_dataset import IP102Dataset
from utils.train_utils import HistoryTracker, save_checkpoint, print_epoch_metrics
from utils.plot_utils import plot_history
from utils.experiment_utils import create_experiment_dir


# Paths
EXP_DIR = create_experiment_dir("experiments", "Baseline")
METADATA = "data/ip102/metadata.csv"
IMAGES = "data/ip102/images"
CHECKPOINT_PATH = f"{EXP_DIR}/checkpoint.pth"
HISTORY_PATH = f"{EXP_DIR}/history.json"
PLOTS_DIR = f"{EXP_DIR}/plots"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset
train_dataset = IP102Dataset(METADATA, IMAGES, split="train", transform=train_transform)
val_dataset = IP102Dataset(METADATA, IMAGES, split="val", transform=val_transform)

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# Model
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 102)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 30
tracker = HistoryTracker()

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100.0 * correct / total

    metrics = {
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_acc": val_acc,
    }
    tracker.update(metrics)
    print_epoch_metrics(epoch, EPOCHS, metrics)

save_checkpoint(model, CHECKPOINT_PATH)
tracker.save_json(HISTORY_PATH)
plot_history(tracker.get(), PLOTS_DIR, prefix="baseline")