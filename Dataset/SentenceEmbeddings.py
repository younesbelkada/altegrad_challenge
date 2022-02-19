from random import randint
import os
import numpy as np
import torch
import random

from Dataset.BaseSentenceEmbeddings import BaseSentenceEmbeddings
import torch.nn.functional as F

from utils.dataset_utils import get_authors_dict


class SentenceEmbeddingsGraphAbstractTwo(BaseSentenceEmbeddings):
    def __init__(self, params) -> None:
        super().__init__(params, "SentenceEmbeddingsGraphAbstractTwo")

    def load_abstract_embeddings(self):
        list_name_em = ['artifacts/Allenai-SpecterEmbedding:v1/embeddings.npy', 
                        'artifacts/scibert_scivocab_uncased.npy:v0/scibert_scivocab_uncased.npy']
        self.logger.info("Loading embeddings directly")

        self.logger.info(f"Loading {list_name_em}")

        self.embeddingsSpecter = np.load(open(list_name_em[0], 'rb')) 
        self.embeddingsScibert = np.load(open(list_name_em[1], 'rb')) 

    def __getitem__(self, idx):
        
        emb1spec = torch.from_numpy(self.embeddingsSpecter[int(self.X[idx, 0])])
        emb2spec = torch.from_numpy(self.embeddingsSpecter[int(self.X[idx, 1])])

        emb1sci = torch.from_numpy(self.embeddingsScibert[int(self.X[idx, 0])])
        emb2sci = torch.from_numpy(self.embeddingsScibert[int(self.X[idx, 1])])
        
        concatenated_embeddings = torch.cat((emb1spec, emb2spec, emb1sci, emb2sci, torch.Tensor(self.X[idx, 2:])), dim=0)

        if not self.predict_mode:
            label = self.y[idx]
            return concatenated_embeddings, label
        return concatenated_embeddings

class SentenceEmbeddingsFeatures(BaseSentenceEmbeddings):
    def __init__(self, params) -> None:
        super().__init__(params, "SentenceEmbeddingsFeatures")
        
    def __getitem__(self, idx):
        
        n1, n2 = self.X[idx]

        concatenated_embeddings = self.get_final_embeddings(n1, n2)
        
        if not self.predict_mode:
            label = self.y[idx]
            return concatenated_embeddings, label
        return concatenated_embeddings