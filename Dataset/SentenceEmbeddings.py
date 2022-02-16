from pyexpat.errors import XML_ERROR_CANT_CHANGE_FEATURE_ONCE_PARSING
import os
from random import randint
import os.path as osp

import torch
import networkx as nx
import numpy as np
import logging
from torch.nn.utils.rnn import pad_sequence
from sentence_transformers import SentenceTransformer
from torch_geometric.utils.convert import from_scipy_sparse_matrix

from utils.logger import init_logger
from torch.utils.data import Dataset
import torch.nn.functional as F

from utils.dataset_utils import get_specter_abstracts_dict

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
                X.append((int(line[0]), int(line[1])))
        self.X = np.array(X)
        self.y = np.zeros(self.X.shape[0])

    def build_train(self):
        '''
        https://huggingface.co/sentence-transformers/allenai-specter
        https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1
        https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1
        '''
        self.predict_mode = False
        
        if self.params.force_create:
            self.logger.info("Force create enabled, creating the embeddings from scratch")

            model = SentenceTransformer(self.params.name_sentence_transformer).to(self.device)
            embeddings = []

            i = 0
            while i < len(self.abstracts):
                abstracts_batch = [self.abstracts[abstract_id] for abstract_id in list(self.abstracts.keys())[i:min(i+self.params.batch_size, len(self.abstracts))]]
                sentence_level_embeddings = model.encode(abstracts_batch, convert_to_numpy = True, show_progress_bar=False)
                doc_level_embedding = sentence_level_embeddings
                embeddings.extend(doc_level_embedding)
                i += self.params.batch_size
            
            self.embeddings = np.array(embeddings)
            name_file = osp.join(os.getcwd(), "input", self.params.name_sentence_transformer.split("/")[-1] + ".npy")
            np.save(open(name_file, "wb"), self.embeddings)
        elif os.path.isfile(self.embeddings_file):
            self.logger.info("Embedings file already exists, loading it directly")
            self.logger.info(f"Loading {self.embeddings_file}")
            self.embeddings = np.load(open(self.embeddings_file, 'rb'))

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

            X_train[2*i+1] = (n1, n2)
            y_train[2*i+1] = 0

        self.X = X_train
        self.y = y_train    
        
    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        emb1 = torch.from_numpy(self.embeddings[int(self.X[idx, 0])])
        emb2 = torch.from_numpy(self.embeddings[int(self.X[idx, 1])])
        concatenated_embeddings = torch.cat((emb1, emb2), dim=0)
        if not self.predict_mode:
            label = self.y[idx]
            return concatenated_embeddings, label
        return concatenated_embeddings


class SentenceEmbeddingsGraph(BaseSentenceEmbeddings):
    def __init__(self, params) -> None:
        super().__init__(params, "SentenceEmbeddingsGraph")

    def __getitem__(self, idx):
        emb1 = torch.from_numpy(self.embeddings[int(self.X[idx, 0])])
        emb2 = torch.from_numpy(self.embeddings[int(self.X[idx, 1])])
    
        concatenated_embeddings = torch.cat((emb1, emb2, torch.Tensor([self.X[idx,2], self.X[idx,3], self.X[idx,4], self.X[idx,5]])), dim=0)

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