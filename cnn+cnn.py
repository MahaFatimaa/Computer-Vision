import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import pandas as pd
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
dataset_dir = 'data'
train_dir = os.path.join(dataset_dir, 'train')
valid_dir = os.path.join(dataset_dir, 'valid')
test_dir = os.path.join(dataset_dir, 'test')
checkpoint_path = "best_model.pth"
csv_log_path = "training_log.csv"

# Transforms
transform = Compose([
    Resize(64),
    CenterCrop(64),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Datasets and Dataloaders
train_dataset = ImageFolder(train_dir, transform=transform)
valid_dataset = ImageFolder(valid_dir, transform=transform)
test_dataset = ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=4)
test_loader = DataLoader(test_dataset, batch_size=4)

num_classes = len(train_dataset.classes)

# CNNModel2
class CNNModel2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Initialize
model = CNNModel2(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Resume
best_accuracy = 0.0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_accuracy = checkpoint['best_accuracy']
    print(f"Resumed from checkpoint with best accuracy: {best_accuracy:.4f}")

# Training
def train_model(model, train_loader, valid_loader, epochs=10):
    global best_accuracy
    history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_acc = validate_model(model, valid_loader)
        avg_loss = running_loss / len(train_loader)

        history.append({
            "epoch": int(epoch+1),
            "loss": float(avg_loss),
            "val_accuracy": float(val_acc)
        })

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy
            }, checkpoint_path)
            print(f"New best model saved (accuracy: {best_accuracy:.4f})")

    # Save log
    try:
        df = pd.DataFrame(history)
        df.to_csv(csv_log_path, index=False)
        print(f"Training log saved to {csv_log_path}")
        plot_training_history(df)
    except Exception as e:
        print(f"Logging error: {e}")

# Validation
def validate_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    print(f"✅ Validation Accuracy: {acc:.4f}")
    return acc

# Evaluation
def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print("\n🧪 Test Results:")
    print(f"Accuracy:  {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Precision: {precision_score(all_labels, all_preds, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(all_labels, all_preds, average='weighted'):.4f}")
    print(f"F1 Score:  {f1_score(all_labels, all_preds, average='weighted'):.4f}")

# Plotting
def plot_training_history(df):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["loss"], label="Training Loss")
    plt.plot(df["epoch"], df["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.close()
    print("📊 Saved plot as training_plot.png")

# Execute
train_model(model, train_loader, valid_loader, epochs=2)
evaluate(model, test_loader)
