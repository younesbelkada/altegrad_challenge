import torch.nn as nn
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning import LightningModule
from torch.nn import functional as F
import torch
from utils.agent_utils import get_net

class BaseModule(LightningModule):
    def __init__(self, network_param, optim_param):
        """method used to define our model parameters
        """
        super().__init__()
        
        # loss function
        self.loss = nn.CrossEntropyLoss()

        self.optim_param = optim_param
        self.lr = optim_param.lr
            
        # get model
        self.model = get_net(network_param, network_param)
        if network_param.weight_checkpoint != "":
            self.model.load_state_dict(torch.load(network_param.weight_checkpoint)["state_dict"])
        
    def forward(self, x, Adj = None):
        raise NotImplementedError(f'Should be implemented in derived class!')

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

    def configure_optimizers(self):
        """defines model optimizer"""
        optimizer = getattr(torch.optim,self.optim_param.optimizer)
        optimizer = optimizer(self.parameters(), lr=self.optim_param.lr)
        
        if self.optim_param.scheduler:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=5, max_epochs=40
            )
            return [[optimizer], [scheduler]]

        return optimizer 


    def _get_loss(self, batch):
        """convenience function since train/valid/test steps are similar"""
        x1, x2 = batch
        z1, z2 = self(x1, x2)

        loss = self.loss(z1, z2)

        return loss