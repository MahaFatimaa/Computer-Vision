import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transforms
transform = Compose([
    Resize(224),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Datasets
dataset_dir = 'data'
train_dataset = ImageFolder(os.path.join(dataset_dir, 'train'), transform=transform)
val_dataset   = ImageFolder(os.path.join(dataset_dir, 'valid'), transform=transform)
test_dataset  = ImageFolder(os.path.join(dataset_dir, 'test'), transform=transform)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

num_classes = len(train_dataset.classes)
print(f"Detected {num_classes} classes.")

# --------------------------------------------
# TransformerBlock (Figure 3b)
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

# Block II = two Transformer blocks
class BlockII(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        self.block1 = TransformerBlock(embed_dim, num_heads)
        self.block2 = TransformerBlock(embed_dim, num_heads)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

# Final Model
class TransformerModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.patch_proj = nn.Conv2d(3, 256, kernel_size=16, stride=16)  # 224x224 → 14x14
        self.pos_embed = nn.Parameter(torch.randn(1, 14*14, 256))
        self.encoder = BlockII(embed_dim=256, num_heads=4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(14*14*256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.patch_proj(x)  # [B, 256, 14, 14]
        x = x.flatten(2).transpose(1, 2)  # [B, 196, 256]
        x = x + self.pos_embed
        x = self.encoder(x)  # Block II
        return self.classifier(x)

model = TransformerModel(num_classes).to(device)

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
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

# Evaluation
def evaluate(loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average='weighted'),
        "precision": precision_score(all_labels, all_preds, average='weighted'),
        "recall": recall_score(all_labels, all_preds, average='weighted'),
    }

# Run training
for epoch in range(30):
    loss = train_epoch(train_loader)
    metrics = evaluate(val_loader)
    print(f"Epoch {epoch+1} | Loss: {loss:.4f} | "
          f"Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")

# Final test
print("\nEvaluating on test set...")
test_metrics = evaluate(test_loader)
print("Test Results:")
for k, v in test_metrics.items():
    print(f"{k.capitalize()}: {v:.4f}")
