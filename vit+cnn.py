import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transforms
transform = Compose([
    Resize(224),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Dataset loading
dataset_dir = 'data'
train_dataset = ImageFolder(os.path.join(dataset_dir, 'train'), transform=transform)
val_dataset   = ImageFolder(os.path.join(dataset_dir, 'valid'), transform=transform)
test_dataset  = ImageFolder(os.path.join(dataset_dir, 'test'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

num_classes = len(train_dataset.classes)
print(f"Detected {num_classes} classes.")

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

# CNN Block (from paper, no padding)
class CNNBlock(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self.block(x)

# Block IV = Transformer → CNN
class BlockIV(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        self.transformer = TransformerBlock(embed_dim, num_heads)
        self.cnn = CNNBlock(in_channels=embed_dim, out_channels=256)

    def forward(self, x):
        x = self.transformer(x)                       # [B, 196, 256]
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, 14, 14)   # → [B, 256, 14, 14]
        x = self.cnn(x)                               # → [B, 256, H, W]
        return x

# Full Model using Block IV
class TransformerCNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.patch_proj = nn.Conv2d(3, 256, kernel_size=16, stride=16)  # → [B, 256, 14, 14]
        self.pos_embed = nn.Parameter(torch.randn(1, 196, 256))
        self.block_iv = BlockIV(embed_dim=256, num_heads=4)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # → [B, 256, 1, 1]
            nn.Flatten(),                 # → [B, 256]
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.patch_proj(x)                       # [B, 256, 14, 14]
        x = x.flatten(2).transpose(1, 2)             # → [B, 196, 256]
        x = x + self.pos_embed
        x = self.block_iv(x)                         # [B, 256, H, W]
        x = self.classifier(x)                       # [B, num_classes]
        return x

# Instantiate model
model = TransformerCNNModel(num_classes).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training and evaluation
def train_epoch(loader):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='weighted'),
        "precision": precision_score(labels, preds, average='weighted'),
        "recall": recall_score(labels, preds, average='weighted'),
    }

# Training loop
for epoch in range(10):
    loss = train_epoch(train_loader)
    metrics = evaluate(val_loader)
    print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")

# Final test evaluation
print("\nFinal Test Evaluation:")
test_metrics = evaluate(test_loader)
for k, v in test_metrics.items():
    print(f"{k.capitalize()}: {v:.4f}")
