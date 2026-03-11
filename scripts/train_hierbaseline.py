import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.ip102_hier_dataset import IP102HierDataset

METADATA = "data/ip102/metadata_hierarchical.csv"
IMAGES = "data/ip102/images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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

train_dataset = IP102HierDataset(METADATA, IMAGES, split="train", transform=train_transform)
val_dataset = IP102HierDataset(METADATA, IMAGES, split="val", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))

# number of classes
num_label_classes = 102
num_genus_classes = train_dataset.df["genus_id"].nunique()
num_family_classes = train_dataset.df["family_id"].nunique()
num_order_classes = train_dataset.df["order_id"].nunique()

print("Genus classes:", num_genus_classes)
print("Family classes:", num_family_classes)
print("Order classes:", num_order_classes)

class HierResNet(nn.Module):
    def __init__(self, num_label_classes, num_genus_classes, num_family_classes, num_order_classes):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
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

model = HierResNet(
    num_label_classes=num_label_classes,
    num_genus_classes=num_genus_classes,
    num_family_classes=num_family_classes,
    num_order_classes=num_order_classes,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        genus_ids = batch["genus_id"].to(device, non_blocking=True)
        family_ids = batch["family_id"].to(device, non_blocking=True)
        order_ids = batch["order_id"].to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)

        loss_label = criterion(outputs["label"], labels)
        loss_genus = criterion(outputs["genus"], genus_ids)
        loss_family = criterion(outputs["family"], family_ids)
        loss_order = criterion(outputs["order"], order_ids)

        loss = loss_label + 0.5 * loss_genus + 0.3 * loss_family + 0.2 * loss_order
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    correct_label = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            genus_ids = batch["genus_id"].to(device, non_blocking=True)
            family_ids = batch["family_id"].to(device, non_blocking=True)
            order_ids = batch["order_id"].to(device, non_blocking=True)

            outputs = model(images)

            loss_label = criterion(outputs["label"], labels)
            loss_genus = criterion(outputs["genus"], genus_ids)
            loss_family = criterion(outputs["family"], family_ids)
            loss_order = criterion(outputs["order"], order_ids)

            loss = loss_label + 0.5 * loss_genus + 0.3 * loss_family + 0.2 * loss_order
            val_loss += loss.item()

            preds = outputs["label"].argmax(dim=1)
            correct_label += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100.0 * correct_label / total

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Label Acc: {val_acc:.2f}%"
    )

os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/resnet18_ip102_hierarchical.pth")
print("Saved checkpoint: checkpoints/resnet18_ip102_hierarchical.pth")