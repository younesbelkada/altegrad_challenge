from sklearn.utils import shuffle
from torch.utils.data import DataLoader

from Dataset.dataset import BaselineGraphDataset
from Model.LogisticRegression import LogisticRegression

def get_train_dataset(params):
    train_dataloader = None
    if params.hparams.dataset == 'BaselineGraphDataset':
        train_dataset = BaselineGraphDataset(params)
        train_dataset.build_train()
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=params.data_param.batch_size, num_workers=params.data_param.num_workers)

    return train_dataloader

def get_model(params):
    model = None
    if params.hparams.model_type == 'LogisticRegression':
        model = LogisticRegression()
    return model
