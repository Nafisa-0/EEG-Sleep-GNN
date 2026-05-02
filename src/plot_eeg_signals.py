import torch
import matplotlib.pyplot as plt
import random

from dataset import EEGGraphDataset

# Path
GRAPH_PATH = r"D:\EEG-Sleep-GNN\graphs\processed"

# Load dataset
dataset = EEGGraphDataset(GRAPH_PATH)

# Find one Wake (0) and one N3 (3)
wake_sample = None
n3_sample = None

for data in dataset:
    if data.y.item() == 0 and wake_sample is None:
        wake_sample = data
    if data.y.item() == 3 and n3_sample is None:
        n3_sample = data
    if wake_sample and n3_sample:
        break

# Extract features (use first channel)
wake_signal = wake_sample.x[0].cpu().numpy()
n3_signal = n3_sample.x[0].cpu().numpy()

# Plot
plt.figure(figsize=(10,5))

plt.subplot(2,1,1)
plt.plot(wake_signal)
plt.title("Wake (W) EEG Signal")

plt.subplot(2,1,2)
plt.plot(n3_signal)
plt.title("Deep Sleep (N3) EEG Signal")

plt.tight_layout()
plt.savefig("eeg_comparison.png", dpi=300)
plt.show()