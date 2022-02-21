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
from Agents.BaseTrainer import BaseTrainer


def main():
    parameters = Parameters.parse()

    # initialize wandb instance
    wdb_config = parse_params(parameters)
    
    if parameters.hparams.train:
        wandb.init(
                # vars(parameters),  # FIXME use the full parameters
                config = wdb_config,
                project = parameters.hparams.wandb_project,
                entity = parameters.hparams.wandb_entity,
                allow_val_change=True,
                job_type="train"
            )
        
        wandb_run = WandbLogger(
            config=wdb_config,# vars(parameters),  # FIXME use the full parameters
            project=parameters.hparams.wandb_project,
            entity=parameters.hparams.wandb_entity,
            allow_val_change=True,
            #save_dir=parameters.hparams.save_dir,
        )
        
        agent = BaseTrainer(parameters, wandb_run)

        if not parameters.data_param.only_create_abstract_embeddings and not parameters.data_param.only_create_keywords :
            agent.run()
            
    else: 
        wandb.init(
                # vars(parameters),  # FIXME use the full parameters
                config = wdb_config,
                project = parameters.hparams.wandb_project,
                entity = parameters.hparams.wandb_entity,
                allow_val_change=True,
                job_type="test"
        )
        agent = BaseTrainer(parameters)
        agent.predict()
        


if __name__ == '__main__':
    main()
