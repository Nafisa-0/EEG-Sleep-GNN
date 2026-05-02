import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATBlock(torch.nn.Module):
    """
    GAT layer + BatchNorm + residual skip connection.
    Safe to stack regardless of dimension changes.
    """

    def __init__(self, in_channels, out_channels, heads=1,
                 concat=True, dropout=0.4):
        super().__init__()

        self.conv = GATConv(in_channels, out_channels,
                            heads=heads, concat=concat, dropout=dropout)

        actual_out = out_channels * heads if concat else out_channels
        self.bn    = torch.nn.BatchNorm1d(actual_out)

        self.res_proj = (
            torch.nn.Linear(in_channels, actual_out, bias=False)
            if in_channels != actual_out else None
        )

    def forward(self, x, edge_index):
        identity = x if self.res_proj is None else self.res_proj(x)
        x = self.conv(x, edge_index)
        x = self.bn(x)
        return F.elu(x + identity)