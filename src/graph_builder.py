import torch
import numpy as np
from torch_geometric.data import Data
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

    edge_index  = torch.tensor(edge_index,  dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    y           = torch.tensor([label],     dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)