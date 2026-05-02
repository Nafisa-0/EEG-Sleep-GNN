import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from layers import GATBlock

<<<<<<< Updated upstream

class SleepGNN(torch.nn.Module):
    """
    GAT sleep stage classifier.
=======
class Model(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.conv1 = GATConv(num_features, 64, heads=4)
        self.conv2 = GATConv(64*4, 128)
        self.conv3 = GATConv(128, 128, heads=1)

        self.lstm = torch.nn.LSTM(128, 128, num_layers=2, batch_first=True, dropout=0.3)
>>>>>>> Stashed changes

    Architecture:
        GATBlock 1: num_features → hidden*heads  (multi-head, concat)
        GATBlock 2: hidden*heads → hidden         (single head)
        GATBlock 3: hidden       → hidden         (single head, added depth)
        Global mean pool
        MLP: hidden → hidden//2 → num_classes
             with BatchNorm and dropout

    No GRU — removed because DataLoader shuffle breaks temporal order.
    """

    def __init__(self, num_features, num_classes,
                 hidden=64, heads=4, dropout=0.3):
        super().__init__()
        self.dropout_p = dropout

        self.block1 = GATBlock(num_features, hidden,
                               heads=heads, concat=True, dropout=dropout)
        self.block2 = GATBlock(hidden * heads, hidden,
                               heads=1, concat=False, dropout=dropout)
        self.block3 = GATBlock(hidden, hidden,
                               heads=1, concat=False, dropout=dropout)

        self.lin1   = torch.nn.Linear(hidden, hidden // 2)
        self.bn_out = torch.nn.BatchNorm1d(hidden // 2)
        self.lin2   = torch.nn.Linear(hidden // 2, num_classes)

    def forward(self, graph_seq):

<<<<<<< Updated upstream
        x = self.block1(x, edge_index)
        x = self.block2(x, edge_index)
        x = self.block3(x, edge_index)

        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.elu(self.bn_out(self.lin1(x)))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        return self.lin2(x)
=======
        x, edge_index, batch = graph_seq.x, graph_seq.edge_index, graph_seq.batch
    
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
    
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
    
        x = F.elu(self.conv3(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
    
        x = global_mean_pool(x, batch)
    
        x = x.unsqueeze(1)
    
        out, _ = self.lstm(x)
        out = out[:, -1, :]
    
        return self.lin(out)
>>>>>>> Stashed changes
