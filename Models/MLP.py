import torch.nn as nn

# FIXME le MLP fait overfit trop vite (j'ai testé avec les embeddings tout seul ça overfit def water)
class MLP(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        in_size = params.embed_dim
        hidden_dim = params.hidden_dim
        self.norm = nn.LayerNorm
        self.fc = nn.Sequential(
            nn.Linear(in_size, hidden_dim),
            nn.ReLU(), 
            self.norm(hidden_dim),
            # nn.Dropout(params.dropout),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # self.norm(hidden_dim),
            nn.Dropout(params.dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            self.norm(hidden_dim//2),
            nn.Dropout(params.dropout),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            self.norm(hidden_dim//4),
            nn.Dropout(params.dropout),
            nn.Linear(hidden_dim//4, 1)
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(in_size, hidden_dim),
        #     nn.ReLU(), 
        #     self.norm(hidden_dim),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     self.norm(hidden_dim),
        #     nn.Linear(hidden_dim, 1)
        # )

    def forward(self, x):
        out = self.fc(x)
        return out.squeeze()