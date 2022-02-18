from random import randint
import os
import numpy as np
import torch
import random

from Dataset.BaseSentenceEmbeddings import BaseSentenceEmbeddings
import torch.nn.functional as F

from utils.dataset_utils import get_authors_dict

class SentenceEmbeddingsVanilla(BaseSentenceEmbeddings):
    def __init__(self, params) -> None:
        super().__init__(params, "SentenceEmbeddings")

    def build_predict(self):

        self.predict_mode = True
        
        X = []
        with open(self.path_predict, 'r') as file:
            for line in file:
                line = line.split(',')
                edg0 = int(line[0])
                edg1 = int(line[1])
                X.append((edg0, edg1))

        self.X = np.array(X)
        self.y = np.zeros(self.X.shape[0])

    def build_train(self):

        self.predict_mode = False

        m = self.G.number_of_edges()
        n = self.G.number_of_nodes()
        X_train = np.zeros((2*m, 2))
        y_train = np.zeros(2*m)
        nodes = list(self.G.nodes())

        for i, edge in enumerate(self.G.edges()):
            
            X_train[2*i] = edge
            y_train[2*i] = 1 

            n1 = nodes[randint(0, n-1)]
            n2 = nodes[randint(0, n-1)]

            while (n1,n2) in  self.G.edges():
                n1 = nodes[randint(0, n-1)]
                n2 = nodes[randint(0, n-1)]
            
            X_train[2*i+1] = (n1, n2)
            y_train[2*i+1] = 0

        self.X = X_train
        self.y = y_train    

    def __getitem__(self, idx):
        emb1 = torch.from_numpy(self.abstract_embeddings[int(self.X[idx, 0])])
        emb2 = torch.from_numpy(self.abstract_embeddings[int(self.X[idx, 1])])
        
        concatenated_embeddings = torch.cat((emb1, emb2), dim=0)

        if not self.predict_mode:
            label = self.y[idx]
            return concatenated_embeddings, label
        return concatenated_embeddings


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