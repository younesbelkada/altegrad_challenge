import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.embedding = nn.Embedding(params.vocab_size+1, params.hidden_dim)
        self.multi_subgraph1 = nn.TransformerEncoderLayer(
            d_model=params.hidden_dim,
            nhead=params.heads,
        )
        self.multi_subgraph2 = nn.TransformerEncoderLayer(
            d_model=params.hidden_dim,
            nhead=params.heads,
            dim_feedforward=1,
        )

        self.mlp = nn.Linear(
            params.hidden_dim*2, 1
        )

    def forward(self, x):
        neighbors_node1, neighbors_node2 = x

        features_neighbors_1 = self.embedding(neighbors_node1)
        features_neighbors_2 = self.embedding(neighbors_node2)

        features_neighbors_1 = self.multi_subgraph1(features_neighbors_1)
        features_neighbors_2 = self.multi_subgraph1(features_neighbors_2)

        features_neighbors_1 = self.multi_subgraph2(features_neighbors_1)
        features_neighbors_2 = self.multi_subgraph2(features_neighbors_2)

        features_neighbors_1 = torch.sum(features_neighbors_1, dim=1)
        features_neighbors_2 = torch.sum(features_neighbors_2, dim=1)

        output = self.mlp(
            torch.cat((features_neighbors_1,  features_neighbors_2), dim=1))

        return output.squeeze()
