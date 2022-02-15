import torch
import torch.nn as nn

from torch.nn import Linear, ReLU
from torch_geometric.nn import Sequential, GCNConv

from Models.BaseModule import BaseModule

class GraphAutoEncoder(BaseModule):
    def __init__(self, params):
        super(self, GraphAutoEncoder).__init__()
        self.params = params.network_parameters
        self.features = nn.Embedding(self.params.vocab_size, self.params.hidden_dim)

        self.encoder = Sequential(
            (GCNConv(self.params.hidden_dim, self.params.hidden_dim), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (GCNConv(self.params.hidden_dim, self.params.hidden_dim), 'x, edge_index -> x'),
            ReLU(inplace=True)
        )
        self.decoder(
            Linear(self.params.hidden_dim, self.params.vocab_size)
        )

        self.activation = nn.Sigmoid()
    def forward(self, x, adj):
        """
            x: LongTensor of indices

        """
        features = self.features(x)
        output = self.encoder(features)
        output = self.decoder(output)
        decoded_adj = self.activation(torch.mm(output, output.t()))
        return decoded_adj