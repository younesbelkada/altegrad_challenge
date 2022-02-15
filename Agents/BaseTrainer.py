import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import wandb
from utils.agent_utils import get_datamodule
from utils.callbacks import AutoSaveModelCheckpoint

from Models.BaseModule import BaseModule

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    RichProgressBar,
)

from utils.agent_utils import get_artifact
from utils.logger import init_logger

class BaseTrainer:
    def __init__(self, config, run = None) -> None:
        self.config = config.hparams
        self.wb_run = run
        self.network_param = config.network_param 
        
        logger = init_logger("BaseTrainer", self.config.log_level)

        logger.info('Loading artifact...')
        self.load_artifact(config.network_param, config.data_param)

        logger.info('Loading Data module...')
        self.datamodule = get_datamodule(
            config.data_param
        )

        logger.info('Loading Model module...')
        self.pl_model = BaseModule(config.network_param, config.optim_param)
    
        wandb.watch(self.pl_model.model)
    
    def run(self):
        trainer = pl.Trainer(
            #logger=self.wb_run,  # W&B integration
            callbacks=self.get_callbacks(),
            gpus=self.config.gpu,  # use all available GPU's
            max_epochs=self.config.max_epochs,  # number of epochs
            check_val_every_n_epoch=self.config.val_freq,
            fast_dev_run=self.config.dev_run,
            accumulate_grad_batches=self.config.accumulate_size,
            log_every_n_steps=1,
        )
        trainer.fit(self.pl_model, datamodule=self.datamodule)

    def load_artifact(self,network_param,data_param):
        network_param.weight_checkpoint = get_artifact(network_param.artifact, type="model")
        data_param.embeddings_file      = get_artifact(data_param.dataset_artifact, type="dataset")
        
    def get_callbacks(self):
        callbacks = [RichProgressBar(), LearningRateMonitor()]
        monitor = "val/loss"
        mode = "min"
        wandb.define_metric(monitor, summary=mode)
        save_top_k = 5
        every_n_epochs = 1
        callbacks += [
            AutoSaveModelCheckpoint #ModelCheckpoint
            (
                config = (self.network_param).__dict__,
                project = self.config.wandb_project,
                entity = self.config.wandb_entity,
                monitor=monitor,
                mode=mode,
                filename="epoch-{epoch:02d}-val_loss={val/loss:.2f}",
                verbose=True,
                dirpath=self.config.weights_path + f"/{str(wandb.run.name)}",
                save_top_k=save_top_k,
                every_n_epochs=every_n_epochs,
                auto_insert_metric_name=False
            )
        ]  # our model checkpoint callback

        return callbacks