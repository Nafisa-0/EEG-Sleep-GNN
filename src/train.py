import os
import torch
import random
import numpy as np
<<<<<<< Updated upstream
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
=======

from torch.utils.data import DataLoader
from dataset import EEGGraphDataset, collate_fn
from models import Model
from config import DATA_PATH, BATCH_SIZE, EPOCHS, NUM_CLASSES, MODEL_SAVE_PATH
from evaluate import evaluate

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ✅ DEFINE MAIN FIRST
def main():

    # -------------------- DATA --------------------
    dataset = EEGGraphDataset(DATA_PATH)

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    train_size = int(0.8 * len(dataset))

    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    train_data = [dataset[i] for i in train_idx]
    test_data = [dataset[i] for i in test_idx]

    train_loader = DataLoader(
        train_data,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )

    test_loader = DataLoader(
        test_data,
        batch_size=8,
        collate_fn=collate_fn,
        num_workers=2
    )

    # -------------------- MODEL --------------------
    sample = dataset[0][0]
    model = Model(sample.x.shape[1], NUM_CLASSES).to(device)

    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
>>>>>>> Stashed changes

    # -------------------- CLASS WEIGHTS --------------------
    label_counts = [0] * NUM_CLASSES
    for seq in train_data:
        label_counts[seq[-1].y.item()] += 1

    total = sum(label_counts)
    weights = [total / (c + 1e-6) for c in label_counts]
    weights = torch.tensor(weights, dtype=torch.float).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

<<<<<<< Updated upstream
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
=======
    # -------------------- TRAIN FUNCTION --------------------
    def train():
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            batch = [g.to(device) for g in batch]

            outputs = []
            labels = []

            for seq in batch:
                out = model(seq)
                outputs.append(out)
                labels.append(seq.y)

            outputs = torch.cat(outputs)
            labels = torch.cat(labels).to(device)

            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    # -------------------- TRAIN LOOP --------------------
    best_acc = 0
    patience = 15
    counter = 0

    for epoch in range(EPOCHS):
        loss = train()

        scheduler.step()

        train_acc, train_f1, _ = evaluate(model, train_loader, device)
        test_acc, test_f1, cm = evaluate(model, test_loader, device)

        print(f"\nEpoch {epoch+1}")
        print(f"Loss: {loss:.4f}")
        print(f"Train Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Test Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("-----")

        if test_acc > best_acc:
            best_acc = test_acc
            counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered ✅")
            break

    print("Training Done ✅")


# ✅ CALL MAIN LAST (VERY IMPORTANT)
if __name__ == "__main__":
    main()
>>>>>>> Stashed changes
