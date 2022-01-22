# Standard libraries
import os
import logging

# For downloading pre-trained models
import urllib.request
from urllib.error import HTTPError

# PyTorch Lightning
import pytorch_lightning as pl

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# PyTorch geometric
# import torch_geometric
# import torch_geometric.data as geom_data
# import torch_geometric.nn as geom_nn

from config.hparams import Parameters
from Agents.main_agent import TrainAgent
from Agents.utils_agent import get_train_dataset

params = Parameters.parse()
logging.basicConfig(level=params.hparams.log_level)

train_data = get_train_dataset(params)
agent = TrainAgent(params)

pl_trainer = pl.Trainer(
    default_root_dir=params.hparams.root_dir, 
    gpus=params.hparams.gpu,
    max_epochs=params.hparams.max_epochs
)
pl_trainer.fit(agent, train_data)
#agent.run()