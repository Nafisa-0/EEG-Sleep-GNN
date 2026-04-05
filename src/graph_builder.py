import os
import mne
import torch
import numpy as np
from torch_geometric.data import Data
from scipy.signal import welch
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PATH ----------------
RAW_PATH = r"D:\EEG-Sleep-GNN\data\raw\sleep-edf-database-expanded-1.0.0\sleep-cassette"
SAVE_PATH = r"D:\EEG-Sleep-GNN\graphs\processed"

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# ---------------- SETTINGS ----------------
EPOCH_DURATION = 30
FS = 100
K = 4  # neighbors for graph

# ---------------- PSD ----------------
def compute_band_power(signal, fs=100):
    freqs, psd = welch(signal, fs=fs, nperseg=fs*2)

    def band(low, high):
        idx = (freqs >= low) & (freqs <= high)
        return np.mean(psd[idx])

    delta = band(0.5, 4)
    theta = band(4, 8)
    alpha = band(8, 13)
    beta = band(13, 30)

    total = delta + theta + alpha + beta + 1e-6

    features = np.array([
        delta / total,
        theta / total,
        alpha / total,
        beta / total
    ])

    return np.log(features + 1e-6)

# ---------------- LABEL MAPPING ----------------
def map_label(desc):
    if "Sleep stage W" in desc:
        return 0
    elif "Sleep stage 1" in desc or "Sleep stage 2" in desc:
        return 1
    elif "Sleep stage 3" in desc or "Sleep stage 4" in desc or "Sleep stage R" in desc:
        return 2
    else:
        return None

# ---------------- FIND MATCHING HYPNOGRAM ----------------
def find_hypnogram(psg_file, all_files):
    base = psg_file.split("-PSG.edf")[0]

    prefix = base[:-1]  

    for f in all_files:
        if f.startswith(prefix) and "Hypnogram" in f:
            return f

    return None

# ---------------- BUILD GRAPH ----------------
def build_graph(psd_features, label):
    x = torch.tensor(psd_features, dtype=torch.float)

    # KNN graph
    sim = np.abs(np.corrcoef(psd_features))

    edge_index = []
    for i in range(sim.shape[0]):
        neighbors = np.argsort(sim[i])[-K:]
        for j in neighbors:
            if i != j:
                edge_index.append([i, j])

    if len(edge_index) == 0:
        return None

    edge_index = torch.tensor(edge_index).t().contiguous()
    y = torch.tensor([label], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)

# ---------------- MAIN ----------------
files = os.listdir(RAW_PATH)
psg_files = [f for f in files if f.endswith("-PSG.edf")]

graph_id = 0

for file in psg_files:
    psg_path = os.path.join(RAW_PATH, file)

    # find matching hypnogram
    hyp_file = find_hypnogram(file, files)

    if hyp_file is None:
        print(f"No hypnogram found for {file}")
        continue

    hyp_path = os.path.join(RAW_PATH, hyp_file)

    print(f"Processing {file} with {hyp_file}...")

    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    annot = mne.read_annotations(hyp_path)
    raw.set_annotations(annot)

    data = raw.get_data()
    events, event_id = mne.events_from_annotations(raw)

    samples_per_epoch = FS * EPOCH_DURATION

    for event in events:
        start = event[0]
        end = start + samples_per_epoch

        if end > data.shape[1]:
            continue

        # get label
        label_desc = list(event_id.keys())[list(event_id.values()).index(event[2])]
        label = map_label(label_desc)

        if label is None:
            continue

        segment = data[:, start:end]

        psd_features = []
        for ch in range(data.shape[0]):
            psd = compute_band_power(segment[ch])
            mean = np.mean(segment[ch])
            std = np.std(segment[ch])
            features = list(psd) + [mean, std]

            psd_features.append(features)

        psd_features = np.array(psd_features)

        psd_features = (psd_features - psd_features.mean(axis=1, keepdims=True)) / (psd_features.std(axis=1, keepdims=True) + 1e-6) #normalization

        graph = build_graph(psd_features, label)

        if graph is None:
            continue

        torch.save(graph, os.path.join(SAVE_PATH, f"graph_{graph_id}.pt"))
        graph_id += 1

print(f"\nTotal graphs created: {graph_id} ✅")
