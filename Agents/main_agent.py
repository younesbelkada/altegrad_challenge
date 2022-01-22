import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim

from Agents.utils_agent import get_train_dataset, get_model

class TrainAgent(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        self.model = get_model(params)
        self.loss_module = nn.BCELoss()

    def forward(self, data):
        x, y = data['x'].float(), data['y'].float()
        x = self.model(x)

        loss = self.loss_module(x.squeeze(1), y)
        return loss

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("val_loss", loss)