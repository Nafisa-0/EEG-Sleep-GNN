import torch
import random
import numpy as np
import os
from collections import defaultdict

from torch_geometric.loader import DataLoader
from dataset import EEGGraphDataset
from models import Model
from config import DATA_PATH, BATCH_SIZE, EPOCHS, NUM_CLASSES, MODEL_SAVE_PATH, LOG_PATH
from evaluate import evaluate
from utils import setup_logger

# -------------------- SEED --------------------
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# -------------------- LOGGER --------------------
setup_logger(LOG_PATH)

# -------------------- DEVICE --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------- LOAD DATA --------------------
dataset = EEGGraphDataset(DATA_PATH)

# 🔥 -------------------- BALANCE DATA (IMPORTANT) --------------------
indices = list(range(len(dataset)))
random.shuffle(indices)

train_size = int(0.8 * len(dataset))

train_indices = indices[:train_size]
test_indices = indices[train_size:]

train_data = [dataset[i] for i in train_indices]
test_data = [dataset[i] for i in test_indices]

# -------------------- DATALOADER --------------------
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

# -------------------- MODEL --------------------
sample = dataset[0]
model = Model(num_features=sample.x.shape[1], num_classes=NUM_CLASSES).to(device)

# -------------------- LOSS --------------------
criterion = torch.nn.CrossEntropyLoss()

# -------------------- OPTIMIZER --------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# -------------------- TRAIN FUNCTION --------------------
def train():
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)

        optimizer.zero_grad()
        out = model(data)

        loss = criterion(out, data.y)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    return total_loss

# -------------------- TRAIN LOOP --------------------
best_test_acc = 0

for epoch in range(1, EPOCHS + 1):
    loss = train()

    train_acc, train_f1, _ = evaluate(model, train_loader, device)
    test_acc, test_f1, cm = evaluate(model, test_loader, device)

    scheduler.step()

    print(f"\nEpoch {epoch}")
    print(f"Loss: {loss:.4f}")
    print(f"Train Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
    print(f"Test Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("-----")

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

print("\nTraining Complete ✅")
print(f"Best Test Accuracy: {best_test_acc:.4f}")