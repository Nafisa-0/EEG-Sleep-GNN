import os
import torch
<<<<<<< Updated upstream
from torch_geometric.data import Dataset


class EEGGraphDataset(Dataset):
    """
    Loads pre-built PyTorch Geometric graph files (.pt) from disk.
    Files are sorted so their order is stable and reproducible.

    Each graph has:
        x          : (3, 11)  node features (11 per channel)
        edge_index : (2, E)   connectivity
        edge_attr  : (E,)     edge weights
        y          : (1,)     sleep stage label 0–4
    """

    def __init__(self, root):
        super().__init__()
        self.files = sorted([
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.endswith(".pt")
        ])
=======
from torch_geometric.data import Batch

class EEGGraphDataset:
    def __init__(self, root_dir, window=5):   # 🔥 increased window
        self.root_dir = root_dir
        self.files = sorted([f for f in os.listdir(root_dir) if f.endswith('.pt')])
        self.window = window
>>>>>>> Stashed changes

    def __len__(self):
        return len(self.files) - self.window

    def __getitem__(self, idx):
<<<<<<< Updated upstream
        # weights_only=False required for torch_geometric Data objects
        return torch.load(self.files[idx], weights_only=False)
=======
        graphs = []
        for i in range(self.window):
            path = os.path.join(self.root_dir, self.files[idx + i])
            g = torch.load(path, weights_only=False)
            graphs.append(g)

        return graphs


# 🔥 NEW COLLATE FUNCTION (CRITICAL FIX)
def collate_fn(batch):
    batched_sequences = []

    for seq in batch:
        batched_seq = Batch.from_data_list(seq)  # creates .batch properly
        batched_sequences.append(batched_seq)

    return batched_sequences
>>>>>>> Stashed changes
