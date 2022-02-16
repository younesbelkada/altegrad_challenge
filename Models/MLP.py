import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        in_size = params.embed_dim
        hidden_dim = params.hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(in_size*2, hidden_dim),
            nn.ReLU(), 
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(params.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(params.dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim//2),
            nn.Dropout(params.dropout),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim//4),
            nn.Dropout(params.dropout),
            nn.Linear(hidden_dim//4, 1)
        )

    def forward(self, x):
        out = self.fc(x)
        return out.squeeze()