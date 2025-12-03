import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNSenior(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super().__init__()
        self.conv1 = GCNConv(obs_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.fc_out(x)

    def act(self, obs_graph):
        x, edge_index = obs_graph
        with torch.no_grad():
            logits = self.forward(x, edge_index)
        return torch.argmax(logits).item()
