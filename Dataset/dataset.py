from pyexpat.errors import XML_ERROR_CANT_CHANGE_FEATURE_ONCE_PARSING
import os
from random import randint
import os.path as osp

import torch
import networkx as nx
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

from utils.logger import init_logger
from torch.utils.data import Dataset

def get_specter_abstracts_dict(path = '/content/CitationPredictionChallenge/Assets/abstracts.txt'):
    abstracts = dict()
    with open(path, 'r', encoding="latin-1") as f:
        for line in f:
            node, full_abstract = line.split('|--|')
            abstract_list = full_abstract.split('.')
            full_abstract = '[SEP]'.join(abstract_list)

            abstracts[int(node)] = full_abstract
    return abstracts

class BaselineGraphDataset(object):
    def __init__(self, params):
        """
            :- params -: Parameters object
        """
        logging.info("Loading the Graph object from the edgelist")
        path_edges = osp.join(params.root_dataset, 'edgelist.txt')
        self.G = nx.read_edgelist(path_edges, delimiter=',', create_using=nx.Graph(), nodetype=int)
        
    def build_train(self):
        logging.info("Building the training dataset using the nx graph")
        nodes = list(self.G.nodes())
        n = self.G.number_of_nodes()
        m = self.G.number_of_edges()

        X_train = np.zeros((2*m, 2))
        y_train = np.zeros(2*m)
        for i,edge in enumerate(self.G.edges()):
            # an edge
            X_train[2*i,0] = self.G.degree(edge[0]) + self.G.degree(edge[1])
            X_train[2*i,1] = abs(self.G.degree(edge[0]) - self.G.degree(edge[1]))
            y_train[2*i] = 1

            # a randomly generated pair of nodes
            n1 = nodes[randint(0, n-1)]
            n2 = nodes[randint(0, n-1)]
            X_train[2*i+1,0] = self.G.degree(n1) + self.G.degree(n2)
            X_train[2*i+1,1] = abs(self.G.degree(n1) - self.G.degree(n2))
            y_train[2*i+1] = 0

        logging.info('Size of training matrix: {}'.format(X_train.shape))
        self.X = X_train
        self.y = y_train

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(np.array(self.y[idx]))
        output_dict = {'x':x, 'y':y}
        return output_dict

    def __len__(self):
        return self.y.shape[0]

class SpecterEmbeddings(Dataset):
    def __init__(self, params) -> None:
        super().__init__()

        self.logger = init_logger("SpecterEmbeddings", "INFO")

        self.logger.info("Loading the Graph object from the edgelist and the embeddings obtained using the abstracts...")
        path_txt = osp.join(params.root_dataset, 'abstracts.txt')
        path_edges = osp.join(params.root_dataset, 'edgelist.txt')
        self.path_predict = osp.join(params.root_dataset, 'test.txt')
        self.G = nx.read_edgelist(path_edges, delimiter=',', create_using=nx.Graph(), nodetype=int)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.abstracts = get_specter_abstracts_dict(path = path_txt)
        self.params = params
        self.embeddings_file = self.params.embeddings_file

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
        '''
        self.predict_mode = False
        model = SentenceTransformer('sentence-transformers/allenai-specter').to(self.device)
        embeddings = []
        labels = []
        
        if self.params.force_create:
            self.logger.info("Force create enabled, creating the embeddings from scratch")
            i = 0
            while i < len(self.abstracts):
                abstracts_batch = [self.abstracts[abstract_id] for abstract_id in list(self.abstracts.keys())[i:min(i+self.params.batch_size, len(self.abstracts))]]
                sentence_level_embeddings = model.encode(abstracts_batch, convert_to_numpy = True, show_progress_bar=False)
                doc_level_embedding = sentence_level_embeddings
                embeddings.extend(doc_level_embedding)
                i += self.params.batch_size
            
            self.embeddings = np.array(embeddings)
            np.save(open(self.embeddings_file, "wb"), self.embeddings)
        elif os.path.isfile(self.embeddings_file):
            self.logger.info("Embedings file already exists, loading it directly")
            self.logger.info(f"Loading {self.embeddings_file}")
            self.embeddings = np.load(open(self.embeddings_file, 'rb'))

        ## Comment créer les labels?
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


class GraphAutoEncoderDataset(Dataset):
    def __init__(self, params) -> None:
        super().__init__()

        self.logger = init_logger("GraphAutoEncoderDataset", "INFO")

        self.logger.info("Loading the Graph object from the edgelist and the embeddings obtained using the abstracts...")
        path_txt = osp.join(params.root_dataset, 'abstracts.txt')
        path_edges = osp.join(params.root_dataset, 'edgelist.txt')
        self.G = nx.read_edgelist(path_edges, delimiter=',', create_using=nx.Graph(), nodetype=int)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.params = params

    def build_train(self):
        self.adj = nx.adjacency_matrix(self.G)
        self.X = list(self.G.nodes())
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.Tensor(self.X[idx]).long()
        return x, self.adj
