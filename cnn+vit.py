import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor, TrainingArguments, Trainer
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
import os
from glob import glob

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Local dataset directory
dataset_dir = 'data'  # Set this to your actual local path if different

train_dir = os.path.join(dataset_dir, 'train')
valid_dir = os.path.join(dataset_dir, 'valid')
test_dir = os.path.join(dataset_dir, 'test')

output_dir = "./model_output"
os.makedirs(output_dir, exist_ok=True)

# Define function to get the latest checkpoint
def get_latest_checkpoint(output_dir):
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    checkpoints = [d for d in checkpoints if os.path.isdir(d)]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return latest_checkpoint

# ViT processor and transforms
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
transforms = Compose([
    Resize(image_processor.size["height"]),
    CenterCrop(image_processor.size["height"]),
    ToTensor(),
    normalize,
])

# Load datasets
train_dataset = ImageFolder(train_dir, transform=transforms)
val_dataset = ImageFolder(valid_dir, transform=transforms)
test_dataset = ImageFolder(test_dir, transform=transforms)

labels = train_dataset.classes
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for idx, label in enumerate(labels)}

# Hybrid CNN + ViT model
class HybridCNNViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.projection = nn.Conv2d(256, 768, kernel_size=1)

        vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_encoder = vit_model.encoder

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.max_position_embeddings = 785
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.max_position_embeddings, 768))
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

        self.layernorm = nn.LayerNorm(768)
        self.cls_head = nn.Linear(768, num_classes)

    def forward(self, x):
        cnn_features = self.cnn(x)
        embeddings = self.projection(cnn_features)
        B, C, H, W = embeddings.shape
        embeddings = embeddings.flatten(2).transpose(1, 2)
        embeddings = self.layernorm(embeddings)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        vit_outputs = self.vit_encoder(embeddings)
        cls_output = vit_outputs[0][:, 0]
        return self.cls_head(cls_output)

model = HybridCNNViT(num_classes=len(labels)).to(device)

# Compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='weighted'),
        "precision": precision_score(labels, preds, average='weighted'),
        "recall": recall_score(labels, preds, average='weighted'),
    }

# Custom collator
def custom_collate_fn(batch):
    images, labels = zip(*batch)
    return {
        "pixel_values": torch.stack(images),
        "labels": torch.tensor(labels)
    }

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="steps",  
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",  
    report_to="none",
    logging_steps=100,
    disable_tqdm=False,
)

# Trainer wrapper
class WrappedModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, pixel_values=None, labels=None):
        logits = self.model(pixel_values)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}

# Initialize trainer
trainer = Trainer(
    model=WrappedModel(model),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=custom_collate_fn,
)

# Check if there's a checkpoint and resume training from it
latest_checkpoint = get_latest_checkpoint(output_dir)

# Train the model, resuming from the latest checkpoint if available
if latest_checkpoint:
    print(f"Resuming training from checkpoint: {latest_checkpoint}")
    trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    print("No checkpoint found, starting from scratch.")
    trainer.train()

# Final evaluation
print("\nRunning final evaluation...")
test_results = trainer.evaluate(test_dataset)

# Show test metrics
print("\nFinal Test Metrics:")
print("-" * 40)
print(f"Accuracy:  {test_results['eval_accuracy']:.4f}")
print(f"F1 Score:  {test_results['eval_f1']:.4f}")
print(f"Precision: {test_results['eval_precision']:.4f}")
print(f"Recall:    {test_results['eval_recall']:.4f}")
print("-" * 40)

# Save final model
trainer.save_model(os.path.join(output_dir, "final_model"))
torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
print("\nTraining complete. Model saved.")
