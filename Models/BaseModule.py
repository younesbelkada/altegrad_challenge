import torch.nn as nn
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning import LightningModule
import torch

from utils.agent_utils import get_net

class BaseModule(LightningModule):
    def __init__(self, network_param, optim_param):
        """method used to define our model parameters
        """
        super(BaseModule, self).__init__()

        # loss function
        self.loss = nn.BCEWithLogitsLoss()

        # optimizer
        self.optim_param = optim_param
        self.lr = optim_param.lr

        # model
        self.model = get_net(network_param.network_name, network_param)
        if network_param.weight_checkpoint is not None:
            self.model.load_state_dict(torch.load(network_param.weight_checkpoint)["state_dict"])
        
    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        """needs to return a loss from a single batch"""
        loss = self._get_loss(batch)

        # Log loss 
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss = self._get_loss(batch)

        # Log loss
        self.log("val/loss", loss)

        return loss

    def predict_step(self, batch, batch_idx):
        
        x = batch
        output = self(x)
        output = torch.sigmoid(output)

        return output

    def configure_optimizers(self):
        """defines model optimizer"""
        optimizer = getattr(torch.optim, self.optim_param.optimizer)
        optimizer = optimizer(self.parameters(), lr=self.optim_param.lr, weight_decay=self.optim_param.weight_decay)
        
        if self.optim_param.scheduler:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=self.optim_param.warmup_epochs, max_epochs=self.optim_param.max_epochs
            )
            return [[optimizer], [scheduler]]

        return optimizer 

    def _get_loss(self, batch):
        """convenience function since train/valid/test steps are similar"""
        x, y = batch
        output = self(x)

        loss = self.loss(output, y)

        return loss