# train_classification.py
# ==============================
# ISIC Skin Lesion Classification
# ==============================

import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score

# ==============================
# Dataset class
# ==============================
class ISICDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.transform = transform
        self.labels = sorted(self.df['diagnosis_1'].unique())
        self.label2idx = {label: idx for idx, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.label2idx[row['diagnosis_1']]
        return img, label

# ==============================
# Main function
# ==============================
def main():

    # Paths
    train_csv = "data/splits/train.csv"
    val_csv   = "data/splits/val.csv"
    test_csv  = "data/splits/test.csv"

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Datasets and DataLoaders
    train_dataset = ISICDataset(train_csv, transform=train_transform)
    val_dataset   = ISICDataset(val_csv, transform=val_transform)
    test_dataset  = ISICDataset(test_csv, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(train_dataset.labels)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 5  # adjust if needed
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {sum(train_losses)/len(train_losses):.4f} | Val Acc: {val_acc:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/resnet18_isic.pth")
    print("✅ Model saved to models/resnet18_isic.pth")

# ==============================
# Run safely on Windows
# ==============================
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  # Windows safe multiprocessing
    main()