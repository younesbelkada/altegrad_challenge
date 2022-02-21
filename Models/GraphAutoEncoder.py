import torch
import torch.nn as nn

from torch.nn import Linear, ReLU
from torch_geometric.nn import Sequential, GCNConv


class GraphAutoEncoder(nn.Module):
    def __init__(self, params):
        super(GraphAutoEncoder, self).__init__()
        self.params = params
        self.features = nn.Embedding(
            self.params.vocab_size, self.params.hidden_dim)

        self.encoder = Sequential('x, edge_index', [
            (GCNConv(self.params.hidden_dim, 64), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (GCNConv(64, 64), 'x, edge_index -> x'),
            ReLU(inplace=True),
            Linear(64, self.params.hidden_dim),
        ])
        self.decoder = Linear(self.params.hidden_dim, self.params.vocab_size)

        self.activation = nn.Sigmoid()

    def forward(self, x):
        """
            x: LongTensor of indices

        """
        features = self.features(x.long())
        output = self.encoder(features)
        output = self.decoder(output)
        decoded_adj = self.activation(torch.mm(output, output.t()))
        return decoded_adj
