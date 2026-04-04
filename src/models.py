import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class Model(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(Model, self).__init__()

        self.conv1 = GATConv(num_features, 64, heads=4)
        self.conv2 = GATConv(64*4, 128, heads=1)

        self.lin = torch.nn.Linear(128, num_classes)
        self.dropout = 0.5

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))

        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.lin(x)