import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
import Dataset


class BaseDataModule(LightningDataModule):
    def __init__(self, dataset_param, dataset_name="BaselineGraphDataset"):
        super().__init__()
        self.dataset = getattr(Dataset.dataset, dataset_name)
        self.config = dataset_param

    def prepare_data(self):
        # use to download
        self.dataset(self.config)
        self.dataset.build_train()
        
    def setup(self, stage=None):
        # transforms
        # split dataset
        val_length = int(len(self.dataset)*self.config.split_val)
        lengths = [len(self.dataset)-val_length, val_length]
        self.train_dataset, self.val_dataset = random_split(self.dataset, lengths)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset, 
            shuffle=True, 
            batch_size  = self.config.batch_size, 
            num_workers = self.config.num_workers
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset, 
            shuffle=False, 
            batch_size  = self.config.batch_size, 
            num_workers = self.config.num_workers
        )
        return val_loader
