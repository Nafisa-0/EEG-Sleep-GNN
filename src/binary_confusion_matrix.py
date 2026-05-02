import torch
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from torch_geometric.loader import DataLoader

from models import SleepGNN
from dataset import EEGGraphDataset

# ------------------------
# CONFIG (FROM YOUR FILE)
# ------------------------
GRAPH_PATH = r"D:\EEG-Sleep-GNN\graphs\processed"
MODEL_PATH = r"D:\EEG-Sleep-GNN\outputs\models\best_model_v1.pt"

BATCH_SIZE = 32
TRAIN_RATIO = 0.8

# ------------------------
# DEVICE
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# LOAD DATASET
# ------------------------
full_dataset = EEGGraphDataset(GRAPH_PATH)

# SAME SPLIT AS TRAINING
from torch.utils.data import Subset

train_size = int(TRAIN_RATIO * len(full_dataset))

indices = list(range(len(full_dataset)))
test_indices = indices[train_size:]

test_dataset = Subset(full_dataset, test_indices)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Total graphs:", len(full_dataset))
print("Test graphs:", len(test_dataset))

# ------------------------
# LOAD MODEL
# ------------------------
from config import HIDDEN, HEADS, DROPOUT, NUM_CLASSES

num_features = full_dataset[0].x.shape[1]

model = SleepGNN(
    num_features=num_features,
    num_classes=NUM_CLASSES,
    hidden=HIDDEN,
    heads=HEADS,
    dropout=DROPOUT
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

print("Model loaded")

# ------------------------
# COLLECT PREDICTIONS
# ------------------------
y_true = []
y_pred = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)

        out = model(data)
        preds = out.argmax(dim=1)

        y_true.extend(data.y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("Predictions collected")

# ------------------------
# FILTER → BINARY (W=0, N3=3)
# ------------------------
y_true_bin = []
y_pred_bin = []

for t, p in zip(y_true, y_pred):
    if t == 0 or t == 3:  # W or N3
        y_true_bin.append(0 if t == 0 else 1)
        y_pred_bin.append(0 if p == 0 else 1)

print("Binary samples:", len(y_true_bin))

# ------------------------
# CONFUSION MATRIX
# ------------------------
labels = ["Wake", "N3"]

ConfusionMatrixDisplay.from_predictions(
    y_true_bin,
    y_pred_bin,
    display_labels=labels,
    xticks_rotation=45
)

plt.tight_layout()
plt.savefig("binary_confusion_matrix.png", dpi=300)
plt.show()

print("Saved as binary_confusion_matrix.png")