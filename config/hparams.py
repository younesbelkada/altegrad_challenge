
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
    debug         : bool = True            # test code before running, if testing, no checkpoints are written
    wandb_project : str  = (f"{'test-'*debug}altegrad")
    root_dir      : str  = os.getcwd()  # root_dir
    seed_everything: Optional[int] = 42   # seed for the whole run
    tune_lr        : bool          = False  # tune the model on first run
    gpu            : int           = 1      # number or gpu
    val_freq       : int           = 1      # validation frequency
    accumulate_size: int           = 1    # gradient accumulation batch size
    max_epochs     : int           = 1000    # maximum number of epochs
    weights_path   : str           = "weights"
    dev_run        : bool          = False
    train          : bool          = False
    best_model     : str           = "deeply-tulip-125"

@dataclass
class NetworkParams:
    network_name   : Optional[str] = "MLP"     # dataset, use <Dataset>Eval for FT
    weight_checkpoints : str = ""
    artifact : str = ""
    vocab_size : int = 138499
    hidden_dim : int = 256
    embed_dim : int = 768

@dataclass
class OptimizerParams: 
    """Optimization parameters"""

    optimizer           : str            = "Adam"  # Optimizer default vit: AdamW, default resnet50: Adam
    lr                  : float          = 3e-4     # learning rate, default = 5e-4
    min_lr              : float          = 5e-6     # min lr reached at the end of the cosine schedule
    scheduler           : bool           = False

@dataclass
class DatasetParams:
    """Dataset Parameters
    ! The batch_size and number of crops should be defined here
    """
    dataset_name   : Optional[str] = "SpecterEmbeddings"     # dataset, use <Dataset>Eval for FT
    num_workers       : int         = 20         # number of workers for dataloadersint
    batch_size        : int         = 256          # batch_size
    split_val         : float       = 0.2
    root_dataset      : Optional[str] = osp.join(os.getcwd(), "input")
    embeddings_file    : str          = osp.join(os.getcwd(), "input", "embeddings.npy")
    force_create      : bool          = False
    dataset_artifact    : str         = 'altegrad-gnn-link-prediction/altegrad/Allenai-SpecterEmbedding:v1'

@dataclass
class Parameters:
    """base options."""
    hparams       : Hparams         = Hparams()
    data_param    : DatasetParams   = DatasetParams()
    network_param : NetworkParams   = NetworkParams()
    optim_param   : OptimizerParams = OptimizerParams()

    def __post_init__(self):
        """Post-initialization code"""
        # Mostly used to set some values based on the chosen hyper parameters
        # since we will use different models, backbones and datamodules
        self.hparams.wandb_project = (f"{'test-'*self.hparams.debug}altegrad") 
        
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