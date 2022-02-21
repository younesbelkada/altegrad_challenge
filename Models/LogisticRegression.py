import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, params):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        return self.sigmoid(x)
