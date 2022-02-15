import torch
import torch.nn as nn
from torch_geometric.nn.conv import GATConv

class GATModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.embedding = nn.Embedding(params.vocab_size+1, params.hidden_dim)
        self.gat_conv_subgraph1 = GATConv(
            in_channels=params.hidden_dim,
            out_channels=params.hidden_dim,
            heads=params.heads,
            concat=False
        )
        self.gat_conv_subgraph2 = GATConv(
            in_channels=params.hidden_dim,
            out_channels=params.hidden_dim,
            heads=params.heads,
            concat=False
        )
        self.mlp = nn.Linear(
            params.hidden_dim*2, 1
        )

    def forward(self, x):
        neighbors_node1, neighbors_node2, adj1, adj2 = x
        # print(neighbors_node1.shape)
        # print(neighbors_node2.shape)
        #neighbors_node1, neighbors_node2 = neighbors_node1[neighbors_node1 != -1], neighbors_node2[neighbors_node2 != -1]
        
        
        outputs = []
        for i in range(neighbors_node1.shape[0]):
            features_neighbors_1 = self.embedding(neighbors_node1[i])
            features_neighbors_2 = self.embedding(neighbors_node2[i])


            features_neighbors_1 = self.gat_conv_subgraph1(features_neighbors_1, adj1[i])
            features_neighbors_2 = self.gat_conv_subgraph2(features_neighbors_2, adj2[i])

            # print(features_neighbors_1.shape)
            # print(features_neighbors_2.shape)

            output = self.mlp(torch.cat((features_neighbors_1[0], features_neighbors_2[0]), dim=0))
            outputs.append(output)
        return torch.cat(outputs)