import torch
import numpy as np
from torch_geometric.data import Data
<<<<<<< Updated upstream
from config import CORR_THRESHOLD, CHANNELS
from utils import extract_features, eog_extra_features

EOG_IDX = CHANNELS.index("EOG horizontal") if "EOG horizontal" in CHANNELS else None

def build_graph(segment, label):
    n_ch = segment.shape[0]
    padded = []
    for ch in range(n_ch):
        base = extract_features(segment[ch])
        extra = eog_extra_features(segment[ch]) if (ch == EOG_IDX and EOG_IDX is not None) else np.zeros(2)
        padded.append(np.concatenate([base, extra]))

    feats = np.array(padded)
    feats = (feats - feats.mean(axis=1, keepdims=True)) / (feats.std(axis=1, keepdims=True) + 1e-6)
    x = torch.tensor(feats, dtype=torch.float)

    corr = np.corrcoef(segment)
    edge_index, edge_weight = [], []
    for i in range(n_ch):
        for j in range(n_ch):
            if i != j and abs(corr[i, j]) > CORR_THRESHOLD:
=======
from scipy.signal import welch

RAW_PATH = r"D:\EEG-Sleep-GNN\data\raw\sleep-edf-database-expanded-1.0.0\sleep-cassette"
SAVE_PATH = r"D:\EEG-Sleep-GNN\graphs\processed"

os.makedirs(SAVE_PATH, exist_ok=True)

EPOCH_DURATION = 30
FS = 100
K = 4

def compute_band_power(signal):
    freqs, psd = welch(signal, fs=FS, nperseg=FS*2)

    def band(low, high):
        idx = (freqs >= low) & (freqs <= high)
        return np.mean(psd[idx])

    delta = band(0.5, 4)
    theta = band(4, 8)
    alpha = band(8, 13)
    beta = band(13, 30)

    total = delta + theta + alpha + beta + 1e-6

    features = np.array([
        delta/total,
        theta/total,
        alpha/total,
        beta/total
    ])

    return np.log(features + 1e-6)

def map_label(desc):
    # ❌ REMOVE WAKE
    if "Sleep stage W" in desc:
        return None

    elif "Sleep stage 1" in desc:
        return 0
    elif "Sleep stage 2" in desc:
        return 1
    elif "Sleep stage 3" in desc or "Sleep stage 4" in desc:
        return 2
    elif "Sleep stage R" in desc:
        return 3

    return None

def build_graph(features, label):
    x = torch.tensor(features, dtype=torch.float)

    from sklearn.metrics.pairwise import cosine_similarity
    
    features = np.nan_to_num(features)

    sim = cosine_similarity(features)

    edge_index = []
    for i in range(sim.shape[0]):
        neighbors = np.argsort(sim[i])[-K:]
        for j in neighbors:
            if i != j:
>>>>>>> Stashed changes
                edge_index.append([i, j])
                edge_weight.append(float(abs(corr[i, j])))

    if len(edge_index) == 0:
        for i in range(n_ch):
            for j in range(n_ch):
                if i != j:
                    edge_index.append([i, j])
                    edge_weight.append(float(abs(corr[i, j])))

    if len(edge_index) == 0:
        return None

<<<<<<< Updated upstream
    edge_index  = torch.tensor(edge_index,  dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    y           = torch.tensor([label],     dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
=======
    edge_index = torch.tensor(edge_index).t().contiguous()
    y = torch.tensor([label], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)

files = os.listdir(RAW_PATH)
psg_files = [f for f in files if f.endswith("-PSG.edf")]

graph_id = 0

for file in psg_files:
    psg_path = os.path.join(RAW_PATH, file)

    base = file.split("-PSG.edf")[0][:-1]
    hyp_file = next((f for f in files if f.startswith(base) and "Hypnogram" in f), None)

    if hyp_file is None:
        continue

    hyp_path = os.path.join(RAW_PATH, hyp_file)

    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    annot = mne.read_annotations(hyp_path)
    raw.set_annotations(annot)

    data = raw.get_data()
    events, event_id = mne.events_from_annotations(raw)

    samples_per_epoch = FS * EPOCH_DURATION

    total_events = len(events)
    for event in events:
        start = event[0]
        end = start + samples_per_epoch

        if end > data.shape[1]:
            continue

        desc = list(event_id.keys())[list(event_id.values()).index(event[2])]
        label = map_label(desc)

        if label is None:
            continue

        segment = data[:, start:end]
        time_feature = start / data.shape[1]

        features = []
        for ch in range(data.shape[0]):
            bp = compute_band_power(segment[ch])

            mean = np.mean(segment[ch])
            std = np.std(segment[ch])

            # 🔥 NEW FEATURES
            skew = np.mean((segment[ch] - mean)**3)
            kurt = np.mean((segment[ch] - mean)**4)

            energy = np.sum(segment[ch]**2)
            
            prob = np.abs(bp)
            prob = prob / (np.sum(prob) + 1e-6)

            entropy = -np.sum(prob * np.log(prob + 1e-6))

            features.append(list(bp) + [mean, std, skew, kurt, energy, entropy, time_feature])

        features = np.array(features)
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        features = (features - features.mean(axis=1, keepdims=True)) / (features.std(axis=1, keepdims=True) + 1e-6)

        graph = build_graph(features, label)
        if graph is None:
            continue

        torch.save(graph, os.path.join(SAVE_PATH, f"graph_{graph_id}.pt"))
        graph_id += 1

print(f"Total graphs created: {graph_id} ✅")
>>>>>>> Stashed changes
