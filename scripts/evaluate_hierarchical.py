import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.ip102_hier_dataset import IP102HierDataset

# Paths
METADATA = "data/ip102/metadata_hierarchical.csv"
IMAGES = "data/ip102/images"
CHECKPOINT = "checkpoints/resnet18_ip102_hierarchical.pth"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Test transforms
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset
test_dataset = IP102HierDataset(
    metadata_csv=METADATA,
    images_dir=IMAGES,
    split="test",
    transform=test_transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

print("Test size:", len(test_dataset))

# Infer taxonomy class counts from metadata
import pandas as pd
df = pd.read_csv(METADATA)
num_label_classes = 102
num_genus_classes = df["genus_id"].nunique()
num_family_classes = df["family_id"].nunique()
num_order_classes = df["order_id"].nunique()

print("Genus classes:", num_genus_classes)
print("Family classes:", num_family_classes)
print("Order classes:", num_order_classes)


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


# Build model
model = HierResNet(
    num_label_classes=num_label_classes,
    num_genus_classes=num_genus_classes,
    num_family_classes=num_family_classes,
    num_order_classes=num_order_classes,
)

# Load checkpoint
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model = model.to(device)
model.eval()

criterion = nn.CrossEntropyLoss()

# Evaluation
test_loss = 0.0
correct = 0
total = 0

# Optional: track taxonomy accuracies too
correct_genus = 0
correct_family = 0
correct_order = 0

with torch.no_grad():
    for batch in test_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        genus_ids = batch["genus_id"].to(device, non_blocking=True)
        family_ids = batch["family_id"].to(device, non_blocking=True)
        order_ids = batch["order_id"].to(device, non_blocking=True)

        outputs = model(images)

        # Main benchmark loss on label head
        loss = criterion(outputs["label"], labels)
        test_loss += loss.item()

        # Main prediction
        preds = outputs["label"].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Optional auxiliary accuracies
        genus_preds = outputs["genus"].argmax(dim=1)
        family_preds = outputs["family"].argmax(dim=1)
        order_preds = outputs["order"].argmax(dim=1)

        correct_genus += (genus_preds == genus_ids).sum().item()
        correct_family += (family_preds == family_ids).sum().item()
        correct_order += (order_preds == order_ids).sum().item()

avg_test_loss = test_loss / len(test_loader)
test_acc = 100.0 * correct / total
genus_acc = 100.0 * correct_genus / total
family_acc = 100.0 * correct_family / total
order_acc = 100.0 * correct_order / total

print(f"Test Loss (label head): {avg_test_loss:.4f}")
print(f"Test Accuracy (label head): {test_acc:.2f}%")
print(f"Genus Accuracy: {genus_acc:.2f}%")
print(f"Family Accuracy: {family_acc:.2f}%")
print(f"Order Accuracy: {order_acc:.2f}%")