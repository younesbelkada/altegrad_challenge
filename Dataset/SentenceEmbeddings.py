import os
from random import randint

import numpy as np
import torch

from Dataset.BaseSentenceEmbeddings import BaseSentenceEmbeddings


class SentenceEmbeddingsVanilla(BaseSentenceEmbeddings):
    def __init__(self, params) -> None:
        super().__init__(params, "SentenceEmbeddings")

    def build_predict(self):

        self.predict_mode = True
        
        if os.path.isfile(self.embeddings_file):
            self.logger.info("Embedings file already exists, loading it directly")
            self.logger.info(f"Loading {self.embeddings_file}")
            self.embeddings = np.load(open(self.embeddings_file, 'rb'))
        else:
            raise NotImplementedError
        
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
        emb1 = torch.from_numpy(self.embeddings[int(self.X[idx, 0])])
        emb2 = torch.from_numpy(self.embeddings[int(self.X[idx, 1])])
        
        concatenated_embeddings = torch.cat((emb1, emb2), dim=0)

        if not self.predict_mode:
            label = self.y[idx]
            return concatenated_embeddings, label
        return concatenated_embeddings


class SentenceEmbeddingsGraphAbstract(BaseSentenceEmbeddings):
    def __init__(self, params) -> None:
        super().__init__(params, "SentenceEmbeddingsGraph")

    def __getitem__(self, idx):
        emb1 = torch.from_numpy(self.embeddings[int(self.X[idx, 0])])
        emb2 = torch.from_numpy(self.embeddings[int(self.X[idx, 1])])
    
        concatenated_embeddings = torch.cat((emb1, emb2, torch.Tensor([self.X[idx, 2:]])), dim=0)

        if not self.predict_mode:
            label = self.y[idx]
            return concatenated_embeddings, label
        return concatenated_embeddings


class SentenceEmbeddingsGraphAbstractTwo(BaseSentenceEmbeddings):
    def __init__(self, params) -> None:
        super().__init__(params, "SentenceEmbeddingsGraph")

    def load_embeddings(self):
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
        
        concatenated_embeddings = torch.cat((emb1spec, emb2spec, emb1sci, emb2sci, torch.Tensor([self.X[idx, 2:]])), dim=0)

        if not self.predict_mode:
            label = self.y[idx]
            return concatenated_embeddings, label
        return concatenated_embeddings


class SentenceEmbeddingsGraphWithNeighbors(BaseSentenceEmbeddings):
    def __init__(self, params) -> None:
        super().__init__(params, "SentenceEmbeddingsGraphWithNeighbors")

    def __getitem__(self, idx):
        current_node1, current_node2 = int(self.X[idx, 0]), int(self.X[idx, 1])

        emb1 = torch.from_numpy(self.embeddings[current_node1])
        emb2 = torch.from_numpy(self.embeddings[current_node2])

        neighbors_node1 = self.G.neighbors(current_node1)
        neighbors_node2 = self.G.neighbors(current_node2)
        
        neighbors_node1 = list(neighbors_node1)
        neighbors_node2 = list(neighbors_node2)
        # Overfit si on met des features avec les voisins, pas les voisins, que voisin train val 
        
        mean_emb_n1 = torch.mean(torch.cat([torch.from_numpy(self.embeddings[n1]).unsqueeze(0) for n1 in neighbors_node1], dim=0), dim=0)
        mean_emb_n2 = torch.mean(torch.cat([torch.from_numpy(self.embeddings[n2]).unsqueeze(0) for n2 in neighbors_node2], dim=0), dim=0)
        
        concatenated_embeddings = torch.cat((emb1, emb2, mean_emb_n1, mean_emb_n2, torch.Tensor([self.X[idx,2], self.X[idx,3], self.X[idx,4], self.X[idx,5]])), dim=0)

        if not self.predict_mode:
            label = self.y[idx]
            return concatenated_embeddings, label
        return concatenated_embeddings
