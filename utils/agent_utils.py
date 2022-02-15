import importlib
from Dataset.datamodule import BaseDataModule
from config.hparams import Parameters
import os
import wandb

def get_net(network_name, network_param):
    """
    Get Network Architecture based on arguments provided
    """
    # FIXME this iss fucking strange the import needs to be done twice to work
    mod = importlib.import_module(f"Models.{network_name}")
    net = getattr(mod,network_name)
    return net(network_param)


def get_artifact(name: str) -> str:
    """Artifact utilities
    Extracts the artifact from the name by downloading it locally>
    Return : str = path to the artifact        
    """
    artifact = wandb.run.use_artifact(name, type='model')
    artifact_dir = artifact.download()
    file_path = os.path.join(artifact_dir, os.listdir(artifact_dir)[0])
    return file_path
    

def get_datamodule(data_param,dataset_name):
    """
    Fetch Datamodule Function Pointer
    """
    return BaseDataModule(data_param,dataset_name)


def parse_params(parameters: Parameters) -> dict:
    wdb_config = {}
    for k,v in vars(parameters).items():
        for key,value in vars(v).items():
            wdb_config[f"{k}-{key}"]=value