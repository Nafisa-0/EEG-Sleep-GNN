import os
import torch
import random
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix
from collections import Counter

from config import (
    GRAPH_PATH, MODEL_SAVE_PATH, VERSION,
    BATCH_SIZE, EPOCHS, NUM_CLASSES, STAGE_NAMES,
    LR, WEIGHT_DECAY, DROPOUT, HIDDEN, HEADS, TRAIN_RATIO,
    NODE_FEAT_DIM, N_NODES
)
from dataset import EEGGraphDataset
from models import SleepGNN
from utils import setup_logger, log


# -----------------------------
# SEED
# -----------------------------
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


# -----------------------------
# LOGGER
# -----------------------------
setup_logger()
log(f"\n{'='*60}")
log(f"VERSION: {VERSION}")
log(f"{'='*60}")


# -----------------------------
# LOAD DATASET
# -----------------------------
log("Loading dataset...")
dataset = EEGGraphDataset(GRAPH_PATH)

labels = [data.y.item() for data in dataset]
print("Label distribution:", Counter(labels))

log(f"Total graphs: {len(dataset)}")


# -----------------------------
# SANITY CHECK
# -----------------------------
actual_nodes = dataset[0].x.shape[0]
actual_dim   = dataset[0].x.shape[1]

if actual_dim != NODE_FEAT_DIM or actual_nodes != N_NODES:
    raise RuntimeError(
        f"Graph shape mismatch: got ({actual_nodes}, {actual_dim}), expected ({N_NODES}, {NODE_FEAT_DIM})"
    )


# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
indices = list(range(len(dataset)))
random.shuffle(indices)

n_train = int(TRAIN_RATIO * len(dataset))

train_idx = indices[:n_train]
test_idx  = indices[n_train:]

train_data = [dataset[i] for i in train_idx]
test_data  = [dataset[i] for i in test_idx]

torch.save(train_idx, "train_idx.pt")
torch.save(test_idx, "test_idx.pt")

log(f"Train: {len(train_data)}  |  Test: {len(test_data)}")


# -----------------------------
# CLASS WEIGHTS
# -----------------------------
all_labels = np.array([g.y.item() for g in train_data])
classes    = np.arange(NUM_CLASSES)

weights = compute_class_weight("balanced", classes=classes, y=all_labels)
weights = np.clip(weights, 0.5, 2.5)

log(f"Class weights: {np.round(weights, 3)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_weights = torch.tensor(weights, dtype=torch.float).to(device)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)


# -----------------------------
# DATALOADER
# -----------------------------
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


# -----------------------------
# MODEL
# -----------------------------
num_features = dataset[0].x.shape[1]

model = SleepGNN(
    num_features=num_features,
    num_classes=NUM_CLASSES,
    hidden=HIDDEN,
    heads=HEADS,
    dropout=DROPOUT
).to(device)


# -----------------------------
# OPTIMIZER
# -----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# -----------------------------
# TRAIN FUNCTION
# -----------------------------
def train_one_epoch():
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

    return total_loss / len(train_loader)


# -----------------------------
# EVALUATE
# -----------------------------
def evaluate(loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data).argmax(dim=1)

            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = (y_true == y_pred).mean()
    f1  = f1_score(y_true, y_pred, average="weighted")
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    return acc, f1, cm


# -----------------------------
# TRAIN LOOP
# -----------------------------
best_f1 = 0
best_acc = 0
best_cm = None

best_train_acc = 0
best_train_f1  = 0

for epoch in range(1, EPOCHS + 1):

    loss = train_one_epoch()
    scheduler.step()

    train_acc, train_f1, _ = evaluate(train_loader)
    test_acc,  test_f1, cm = evaluate(test_loader)

    log(f"\nEpoch {epoch}")
    log(f"Loss: {loss:.4f}")
    log(f"Train Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
    log(f"Test  Acc: {test_acc:.4f}, F1: {test_f1:.4f}")

    # track best train
    if train_f1 > best_train_f1:
        best_train_f1  = train_f1
        best_train_acc = train_acc

    # track best test
    if test_f1 > best_f1:
        best_f1  = test_f1
        best_acc = test_acc
        best_cm  = cm

        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        log("✅ Best model saved")


# -----------------------------
# FINAL RESULTS
# -----------------------------
log("\n" + "="*60)
log("FINAL BEST MODEL RESULTS")
log("="*60)

log(f"Best Train Accuracy: {best_train_acc:.4f}")
log(f"Best Train F1 Score: {best_train_f1:.4f}")

log(f"\nBest Test Accuracy: {best_acc:.4f}")
log(f"Best Test F1 Score: {best_f1:.4f}")

# Per-class accuracy
per_class = best_cm.diagonal() / (best_cm.sum(axis=1) + 1e-8)

for i in range(NUM_CLASSES):
    log(f"{STAGE_NAMES[i]} Accuracy: {per_class[i]*100:.2f}%")

log(f"\nConfusion Matrix:\n{best_cm}")

log("\nTraining Complete")