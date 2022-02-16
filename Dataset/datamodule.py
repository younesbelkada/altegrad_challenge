import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from Dataset import dataset
from Dataset import SentenceEmbeddings



class BaseDataModule(LightningDataModule):
    def __init__(self, dataset_param):
        super().__init__()
        if "SentenceEmbeddings" in dataset_param.dataset_name:
            data_module = getattr(SentenceEmbeddings, dataset_param.dataset_name)
        else:
            data_module = getattr(dataset, dataset_param.dataset_name)

        self.config = dataset_param
        self.dataset = data_module(dataset_param)
        
    def prepare_data(self) -> None:
        return super().prepare_data()
        
    def setup(self, stage=None):
        # Build dataset
        if stage in (None, "fit"):
            # Load dataset
            self.dataset.load_embeddings()
            self.dataset.build_train()
            val_length = int(len(self.dataset)*self.config.split_val)
            lengths = [len(self.dataset)-val_length, val_length]
            self.train_dataset, self.val_dataset = random_split(self.dataset, lengths)

        if stage == "predict":
            self.dataset.build_predict()

    def train_dataloader(self):
        coll_fn = None
        if hasattr(self.dataset,"collate_fn"):
            coll_fn = self.dataset.collate_fn 
        train_loader = DataLoader(
            self.train_dataset, 
            shuffle=True, 
            batch_size  = self.config.batch_size, 
            num_workers = self.config.num_workers,
            collate_fn= coll_fn
        )
        return train_loader

    def val_dataloader(self):
        coll_fn = None
        if hasattr(self.dataset,"collate_fn"):
            coll_fn = self.dataset.collate_fn 
        val_loader = DataLoader(
            self.val_dataset, 
            shuffle = False, 
            batch_size  = self.config.batch_size, 
            num_workers = self.config.num_workers,
            collate_fn=coll_fn
        )
        return val_loader

    def predict_dataloader(self):
        coll_fn = None
        if hasattr(self.dataset,"collate_fn"):
            coll_fn = self.dataset.collate_fn 
        predict_loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle = False,
            collate_fn=coll_fn
        )
        return predict_loader
