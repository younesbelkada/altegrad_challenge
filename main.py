# Standard libraries
import os
import logging
import wandb
# For downloading pre-trained models
import urllib.request
from urllib.error import HTTPError
from pytorch_lightning.loggers import WandbLogger

# PyTorch Lightning
import pytorch_lightning as pl

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.agent_utils import parse_params
from config.hparams import Parameters
#from Agents.main_agent import TrainAgent
from Agents.BaseTrainer import BaseTrainer
from utils.logger import init_logger


def main():
    parameters = Parameters.parse()
    # logging.basicConfig(level=parameters.hparams.log_level)
    logger = init_logger("Main", parameters.hparams.log_level)

    # initialize wandb instance
    wdb_config = parse_params(parameters)

    wandb.init(
        # vars(parameters),  # FIXME use the full parameters
        config=wdb_config,
        project=parameters.hparams.wandb_project,
        entity=parameters.hparams.wandb_entity,
        allow_val_change=True,
    )

    agent = BaseTrainer(parameters)

    # train_data = get_train_dataset(parameters)

    agent.run()


if __name__ == '__main__':
    main()
