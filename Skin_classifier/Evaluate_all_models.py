import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import os
import argparse
import numpy as np

#  Argument Parser 
parser = argparse.ArgumentParser()
parser.add_argument("--test_dir", type=str, default=r"C:\Users\C4\Desktop\A770_FINALL\test_set", help="Path to test dataset")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--cnn_checkpoint", type=str, default=r"checkpoints/custom_model_best.pth")
parser.add_argument("--effnet_checkpoint", type=str, default=r"checkpoints/efficientnet_model_best.pth")
parser.add_argument("--ensemble", action="store_true", help="Evaluate ensemble of CNN + EfficientNet")
args = parser.parse_args()

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Data Transforms & Dataset 
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root=args.test_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
num_classes = len(test_dataset.classes)

# Model Definitions 
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

class EfficientNetWrapper(nn.Module):
    def __init__(self, num_classes=7):
        super(EfficientNetWrapper, self).__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.model(x)

# Test Function 
def test_model(model, test_loader, model_name="Model"):
    model.eval()
    model.to(device)
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print(f"\n📊 {model_name} Test Results:")
    print(classification_report(y_true, y_pred, target_names=test_dataset.classes))

# Ensemble Evaluation 
def ensemble_evaluate(cnn_model, effnet_model, test_loader):
    cnn_model.eval()
    effnet_model.eval()
    cnn_model.to(device)
    effnet_model.to(device)
    
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            cnn_out = cnn_model(images)
            eff_out = effnet_model(images)
            avg_probs = (torch.softmax(cnn_out, dim=1) + torch.softmax(eff_out, dim=1)) / 2
            _, preds = torch.max(avg_probs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\n🔮 Ensemble Test Results:")
    print(classification_report(y_true, y_pred, target_names=test_dataset.classes))

#  Main 
if __name__ == "__main__":
    # Loading CNN model
    cnn_model = SkinCNN(num_classes=num_classes)
    cnn_model.load_state_dict(torch.load(args.cnn_checkpoint, map_location=device, weights_only=True))
    test_model(cnn_model, test_loader, model_name="Custom CNN")

    # Loading EfficientNet model
    effnet_model = EfficientNetWrapper(num_classes=num_classes)
    effnet_model.load_state_dict(torch.load(args.effnet_checkpoint, map_location=device, weights_only=True))

    test_model(effnet_model, test_loader, model_name="EfficientNet")

    # Ensemble evaluation
    if args.ensemble:
        ensemble_evaluate(cnn_model, effnet_model, test_loader)
