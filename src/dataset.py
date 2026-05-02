import os
import torch
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

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # weights_only=False required for torch_geometric Data objects
        return torch.load(self.files[idx], weights_only=False)