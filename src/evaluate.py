import os
import torch
<<<<<<< Updated upstream
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from config import (
    GRAPH_PATH, MODEL_SAVE_PATH, VERSION,
    NUM_CLASSES, STAGE_NAMES, BATCH_SIZE, TRAIN_RATIO,
    HIDDEN, HEADS, DROPOUT
)
from dataset import EEGGraphDataset
from models import SleepGNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset   = EEGGraphDataset(GRAPH_PATH)
train_idx = torch.load("train_idx.pt")
test_idx  = torch.load("test_idx.pt")

test_data = [dataset[i] for i in test_idx]
loader    = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

print(f"Version    : {VERSION}")
print(f"Model path : {MODEL_SAVE_PATH}")
print(f"Test graphs: {len(test_data)}")

num_features = dataset[0].x.shape[1]
model = SleepGNN(num_features=num_features, num_classes=NUM_CLASSES,
                 hidden=HIDDEN, heads=HEADS, dropout=DROPOUT).to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device,
                                 weights_only=True))
model.eval()
print("Model loaded\n")

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
f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
cm  = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

print("=" * 60)
print(f"Overall Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
print(f"Weighted F1      : {f1:.4f}")
print("=" * 60)
print("\nClassification Report:")
print(classification_report(y_true, y_pred,
                             target_names=STAGE_NAMES, zero_division=0))
print("Confusion Matrix (rows=true, cols=pred):")
print("       " + "  ".join(f"{s:>5}" for s in STAGE_NAMES))
for i, row in enumerate(cm):
    print(f"  {STAGE_NAMES[i]:>4} " + "  ".join(f"{v:>5}" for v in row))
print("\nPer-class Accuracy:")
per_class = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
for i, name in enumerate(STAGE_NAMES):
    print(f"  {name}: {per_class[i]:.4f}  ({per_class[i]*100:.1f}%)")
    
    
=======
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def evaluate(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in loader:
            batch = [g.to(device) for g in batch]

            for seq in batch:
                out = model(seq)
                pred = out.argmax(dim=1)

                y_true.extend(seq.y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    return acc, f1, cm
>>>>>>> Stashed changes
