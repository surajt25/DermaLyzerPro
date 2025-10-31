
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import os
import argparse
from sklearn.metrics import classification_report

#  Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
parser.add_argument("--train_cnn", action="store_true", help="Train custom CNN model")
parser.add_argument("--train_effnet", action="store_true", help="Train EfficientNet model")
parser.add_argument("--ensemble_eval", action="store_true", help="Evaluate ensemble of CNN + EfficientNet models")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr_cnn", type=float, default=0.001)
parser.add_argument("--lr_effnet", type=float, default=0.0001)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--cnn_checkpoint", type=str, default=r"checkpoints\custom_model.pth")
parser.add_argument("--effnet_checkpoint", type=str, default=r"checkpoints\efficientnet_model.pth")
args = parser.parse_args()

# Data Transforms & Datasets 
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root="dataset/train", transform=train_transforms)
val_dataset = datasets.ImageFolder(root="dataset/val", transform=val_transforms)

# Weighted sampler for imbalance handling
targets = [label for _, label in train_dataset.samples]
class_counts = np.bincount(targets)
class_weights = 1.0 / (class_counts + 1e-6)
sample_weights = [class_weights[t] for t in targets]
sampler = WeightedRandomSampler(weights=sample_weights,
                                num_samples=len(sample_weights),
                                replacement=True)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

num_classes = len(train_dataset.classes)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")


# Custom CNN Model 
class SkinCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SkinCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# EfficientNet Wrapper
class EfficientNetWrapper(nn.Module):
    def __init__(self, num_classes=7):
        super(EfficientNetWrapper, self).__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.model(x)

# Training loop function with resume support 
def train_model(model, optimizer, scheduler, criterion, epochs, model_name, checkpoint_path):
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    best_acc = 0.0
    patience_counter = 0
    start_epoch = 0

    # Resume logic 
    if args.resume and os.path.exists(checkpoint_path):
        print(f"🔄 Resuming {model_name} training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_acc = checkpoint["best_acc"]
        start_epoch = checkpoint["epoch"] + 1
        patience_counter = checkpoint.get("patience_counter", 0)
        print(f"Resumed from epoch {start_epoch} | Best Val Acc so far: {best_acc:.2f}%")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validation 
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        print(f"[{model_name}] Epoch {epoch+1}/{epochs} "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(classification_report(y_true, y_pred, target_names=val_dataset.classes))

        scheduler.step(val_loss)

        # Saving checkpoint (every epoch) 
        checkpoint_data = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_acc": best_acc,
            "patience_counter": patience_counter,
        }
        torch.save(checkpoint_data, checkpoint_path)

        # Saving best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path.replace(".pth", "_best.pth"))
            print(f"Saved best {model_name} model at {checkpoint_path.replace('.pth', '_best.pth')}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"⏹️ Early stopping {model_name} training")
            break

    print(f"🏁 {model_name} training finished. Best Val Acc: {best_acc:.2f}%")

# = Ensembling
def ensemble_evaluate():
    
    cnn = SkinCNN(num_classes=num_classes).to(device)
    effnet = EfficientNetWrapper(num_classes=num_classes).to(device)

    # using best model versions if they exist
    cnn_ckpt = args.cnn_checkpoint.replace(".pth", "_best.pth") if os.path.exists(args.cnn_checkpoint.replace(".pth", "_best.pth")) else args.cnn_checkpoint
    eff_ckpt = args.effnet_checkpoint.replace(".pth", "_best.pth") if os.path.exists(args.effnet_checkpoint.replace(".pth", "_best.pth")) else args.effnet_checkpoint

    cnn.load_state_dict(torch.load(cnn_ckpt, map_location=device))
    effnet.load_state_dict(torch.load(eff_ckpt, map_location=device))

    cnn.eval()
    effnet.eval()

    val_correct, val_total = 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            cnn_out = cnn(images)
            effnet_out = effnet(images)

            # Averaging probabilities
            avg_probs = (torch.softmax(cnn_out, dim=1) + torch.softmax(effnet_out, dim=1)) / 2
            _, preds = torch.max(avg_probs, 1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    val_acc = 100 * val_correct / val_total
    print(f"\n Ensemble Validation Accuracy: {val_acc:.2f}%")
    print(classification_report(y_true, y_pred, target_names=val_dataset.classes))

# Main
if __name__ == "__main__":
    if args.train_cnn:
        print("Training Custom CNN...")
        cnn_model = SkinCNN(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(cnn_model.parameters(), lr=args.lr_cnn)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        train_model(cnn_model, optimizer, scheduler, criterion,
                    args.epochs, "Custom CNN", args.cnn_checkpoint)

    if args.train_effnet:
        print("Training EfficientNet...")
        effnet_model = EfficientNetWrapper(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(effnet_model.parameters(), lr=args.lr_effnet)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        train_model(effnet_model, optimizer, scheduler, criterion,
                    args.epochs, "EfficientNet", args.effnet_checkpoint)

    if args.ensemble_eval:
        print("Evaluating Ensemble of CNN + EfficientNet...")
        ensemble_evaluate()
