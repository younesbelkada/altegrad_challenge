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

class BaseSentenceEmbeddings(Dataset):
    def __init__(self, params, name_dataset) -> None:
        super().__init__()

        self.logger = init_logger(name_dataset, "INFO")

        self.logger.info("Loading the Graph object from the edgelist and the embeddings obtained using the abstracts...")
        path_txt = osp.join(params.root_dataset, 'abstracts.txt')
        path_edges = osp.join(params.root_dataset, 'edgelist.txt')
        self.path_predict = osp.join(params.root_dataset, 'test.txt')
        self.G = nx.read_edgelist(path_edges, delimiter=',', create_using=nx.Graph(), nodetype=int)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.abstracts = get_specter_abstracts_dict(path = path_txt)
        self.params = params
        self.embeddings_file = self.params.embeddings_file
        self.adj = None

    def get_features(self, edg0, edg1):
        deg_edg0 = self.G.degree(edg0)
        deg_edg1 = self.G.degree(edg1)
        edg_sum = deg_edg0 + deg_edg1
        edg_abs = abs(deg_edg0 - deg_edg1)
        
        return (edg0, edg1, deg_edg0, deg_edg1, edg_sum, edg_abs)

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
                X.append(self.get_features(edg0, edg1))

        self.X = np.array(X)
        self.y = np.zeros(self.X.shape[0])

    def build_train(self):
        '''
        https://huggingface.co/sentence-transformers/allenai-specter
        https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1
        https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1
        https://huggingface.co/allenai/scibert_scivocab_uncased
        '''
        self.predict_mode = False
        
        if self.params.force_create:
            self.logger.info("Force create enabled, creating the embeddings from scratch")
            
            self.logger.info(self.params.name_sentence_transformer)
            model = SentenceTransformer(self.params.name_sentence_transformer).to(self.device)
            embeddings = []
            
            self.logger.info(f"batch size: {self.params.batch_size}")

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
        X_train = np.zeros((2*m, 6))
        y_train = np.zeros(2*m)
        nodes = list(self.G.nodes())
        for i, edge in enumerate(self.G.edges()):

            
            X_train[2*i] = self.get_features(edge[0],edge[1])
            y_train[2*i] = 1 

            n1 = nodes[randint(0, n-1)]
            n2 = nodes[randint(0, n-1)]

            while (n1,n2) in  self.G.edges():
                n1 = nodes[randint(0, n-1)]
                n2 = nodes[randint(0, n-1)]

            X_train[2*i+1] = self.get_features(n1,n2)
            y_train[2*i+1] = 0

        self.X = X_train
        self.y = y_train    
        
    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        raise NotImplementedError(f'Should be implemented in derived class!')