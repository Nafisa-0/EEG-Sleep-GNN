import os
import mne
import numpy as np
from glob import glob
import torch

from config import GRAPH_PATH
from graph_builder import build_graph

# -----------------------------------
# PATH
# -----------------------------------
DREAMS_PATH = r"D:\EEG-Sleep-GNN\data\raw\DatabaseSubjects"
os.makedirs(GRAPH_PATH, exist_ok=True)


# -----------------------------------
# LABEL LOADING (FIXED ✅)
# -----------------------------------
def load_labels(txt_path):
    labels = []

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()

            # Skip header
            if "Hypnogram" in line or line == "":
                continue

            try:
                val = int(line)

                # DREAMS mapping
                if val == 5:       # Wake
                    labels.append(0)

                elif val == 4:     # N3
                    labels.append(3)

                else:
                    labels.append(-1)

            except:
                labels.append(-1)

    return labels


# -----------------------------------
# MAIN
# -----------------------------------
edf_files = glob(os.path.join(DREAMS_PATH, "*.edf"))

print(f"Found {len(edf_files)} DREAMS EDF files")

count = 0
total_epochs = 0
kept_epochs = 0


for edf_file in edf_files:
    try:
        print(f"\nProcessing: {edf_file}")

        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)

        print("CHANNELS:", raw.ch_names)

        # -----------------------------------
        # CHANNEL SELECTION (ROBUST)
        # -----------------------------------
        selected_channels = []

        for ch in raw.ch_names:
            ch_lower = ch.lower()

            if "fp1" in ch_lower:
                selected_channels.append(ch)

            elif "cz" in ch_lower:
                selected_channels.append(ch)

            elif "eog" in ch_lower:
                selected_channels.append(ch)

        selected_channels = list(dict.fromkeys(selected_channels))[:3]

        if len(selected_channels) < 3:
            print("[SKIP] Not enough valid channels")
            continue

        print("Selected channels:", selected_channels)

        raw.pick(selected_channels)

        # -----------------------------------
        # LOAD LABELS
        # -----------------------------------
        base = os.path.basename(edf_file).replace(".edf", "")
        label_file = os.path.join(DREAMS_PATH, f"HypnogramAASM_{base}.txt")

        if not os.path.exists(label_file):
            print(f"[SKIP] Missing label file")
            continue

        labels = load_labels(label_file)

        # -----------------------------------
        # EPOCHING
        # -----------------------------------
        epochs = mne.make_fixed_length_epochs(raw, duration=30, preload=True)

        print(f"Epochs: {len(epochs)}, Labels: {len(labels)}")

        # -----------------------------------
        # PROCESS
        # -----------------------------------
        for i in range(min(len(epochs), len(labels))):

            total_epochs += 1
            y = labels[i]

            # ONLY W vs N3
            if y not in [0, 3]:
                continue

            kept_epochs += 1

            x = epochs[i].get_data().squeeze()

            # Normalize
            x = (x - np.mean(x)) / (np.std(x) + 1e-6)

            graph = build_graph(x, y)

            save_path = os.path.join(GRAPH_PATH, f"dreams_{count}.pt")
            torch.save(graph, save_path)

            count += 1

    except Exception as e:
        print(f"[ERROR] {edf_file} → {e}")


# -----------------------------------
# FINAL OUTPUT
# -----------------------------------
print("\n==============================")
print(f"Total DREAMS graphs created: {count}")
print(f"Total epochs processed     : {total_epochs}")
print(f"Valid W/N3 epochs kept     : {kept_epochs}")
print("==============================")