import torch
import torch.nn as nn
from math import log

class MLPWithAuthors(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        in_size = params.input_size

        self.norm = getattr(nn, params.normalization)
        self.activation = getattr(nn, params.activation)

        self.layers = nn.ModuleList()

        self.authors_embbedings = nn.Embedding(params.nb_authors, params.emb_authors_dim)
        
        self.layers.append(nn.Sequential(nn.Linear(in_size, in_size),
                                            self.norm(in_size),
                                            self.activation(),
                                            nn.Dropout(params.dropout)
                                            )
                            )

        self.layers.append(nn.Sequential(nn.Linear(in_size, in_size),
                                            self.norm(in_size),
                                            self.activation(),
                                            nn.Dropout(params.dropout)
                                            )
                )

        for i in range(int(log(in_size, 2))):
            
                
            self.layers.append(nn.Sequential(nn.Linear(in_size, in_size//2),
                                            self.norm(in_size//2),
                                            self.activation(),
                                            nn.Dropout(params.dropout)
                                            )
            )
            
            in_size //= 2
            
            if in_size <= 400:
                break

        self.layers.append(nn.Linear(in_size, 1))

    def forward(self, x):
        authors, x = x
        authors_node1, authors_node2 = authors
        emb_author1, emb_author2 = self.authors_embbedings(authors_node1), self.authors_embbedings(authors_node2)
        x  = torch.cat((emb_author1, emb_author2, x), dim=1)
        for layer in self.layers:
            x = layer(x)
        return x.squeeze()