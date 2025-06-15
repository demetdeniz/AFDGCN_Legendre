import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP

class Model(nn.Module):
    def __init__(self, num_node, input_dim, hidden_dim, output_dim, **kwargs):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.appnp = APPNP(K=10, alpha=0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.fc1(x))
        x = self.appnp(x, edge_index)
        x = self.fc2(x)
        return x
