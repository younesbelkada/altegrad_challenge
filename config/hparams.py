
import os
import random
from dataclasses import dataclass
from os import path as osp
import logging
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np
import torch
import torch.optim
import pytorch_lightning as pl

import simple_parsing
from simple_parsing.helpers import Serializable, choice, dict_field, list_field

################################## Global parameters ##################################

@dataclass
class Hparams:
    """Hyperparameters of for the run"""

    
    wandb_entity  : str  = "altegrad-gnn-link-prediction"         # name of the project
    test          : bool = False            # test code before running, if testing, no checkpoints are written
    wandb_project : str  = (f"{'test-'*test}altegrad")
    root_dir      : str  = os.getcwd()  # root_dir
    dataset_name   : Optional[str] = "SpecterEmbeddings"     # dataset, use <Dataset>Eval for FT
    network_name   : Optional[str] = "LogisticRegression"     # dataset, use <Dataset>Eval for FT
    log_level       : str             = logging.INFO # Log info level https://docs.python.org/3/howto/logging.html 
    seed_everything: Optional[int] = 42   # seed for the whole run
    tune_lr        : bool          = False  # tune the model on first run
    gpu            : int           = 1      # number or gpu
    precision      : int           = 32     # precision
    val_freq       : int           = 1      # validation frequency
    accumulate_size: int           = 512//64    # gradient accumulation batch size
    max_epochs     : int           = 1000    # maximum number of epochs

@dataclass
class NetworkParams:
    nb_layers : int = 2

@dataclass
class DatasetParams:
    """Dataset Parameters
    ! The batch_size and number of crops should be defined here
    """
    
    num_workers       : int         = 20         # number of workers for dataloadersint
    batch_size        : int         = 2048        # batch_size
    root_dataset      : Optional[str] = osp.join(os.getcwd(), "input")
    force_create      : bool        = True
@dataclass
class Parameters:
    """base options."""
    hparams       : Hparams         = Hparams()
    data_param    : DatasetParams   = DatasetParams()
    network_param
    def __post_init__(self):
        """Post-initialization code"""
        # Mostly used to set some values based on the chosen hyper parameters
        # since we will use different models, backbones and datamodules
        self.hparams.wandb_project = (f"{'test-'*self.hparams.test}altegrad") 
        
        # if "BarlowTwins" in self.hparams.arch:
    
        # Set random seed
        if self.hparams.seed_everything is None:
            self.hparams.seed_everything = random.randint(1, 10000)
            
        random.seed(self.hparams.seed_everything)
        torch.manual_seed(self.hparams.seed_everything)
        pl.seed_everything(self.hparams.seed_everything)

    @classmethod
    def parse(cls):
        parser = simple_parsing.ArgumentParser()
        parser.add_arguments(cls, dest="parameters")
        args = parser.parse_args()
        instance: Parameters = args.parameters
        return instance