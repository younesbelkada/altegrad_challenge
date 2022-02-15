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
        path_edges = osp.join(params.data_param.root_dataset, 'edgelist.txt')
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

class SpecterEmbeddings(object):
    def __init__(self, params) -> None:
        logging.info("Loading the Graph object from the edgelist and the embeddings obtained using the abstracts...")
        path_txt = osp.join(params.data_param.root_dataset, 'abstracts.txt')
        path_edges = osp.join(params.data_param.root_dataset, 'edgelist.txt')
        self.G = nx.read_edgelist(path_edges, delimiter=',', create_using=nx.Graph(), nodetype=int)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.abstracts = get_specter_abstracts_dict(path = path_txt)
        self.params = params.data_param
        self.embeddings_file = self.params.embeddings_file

        self.logger = init_logger("SpecterEmbeddings", params.hparams.log_level)

    def build_train(self):
        '''
        https://huggingface.co/sentence-transformers/allenai-specter
        '''
        model = SentenceTransformer('sentence-transformers/allenai-specter').to(self.device)
        embeddings = []
        labels = []
        if os.path.isfile(self.embeddings_file):
            self.logger.info("Embedings file already exists, loading it directly")
            self.logger.info(f"Loading {self.embeddings_file}")
            self.embeddings = np.load(open(self.embeddings_file, 'rb'))
        elif self.params.force_create:
            self.logger.info("Force create enabled, creating the embeddings from scratch")
            i = 0
            while i < len(self.abstracts):
                abstracts_batch = [self.abstracts[abstract_id] for abstract_id in list(self.abstracts.keys())[i:min(i+self.params.batch_size, len(self.abstracts)-1)]]
                sentence_level_embeddings = model.encode(abstracts_batch, convert_to_numpy = True, show_progress_bar=False)
                doc_level_embedding = sentence_level_embeddings
                embeddings.extend(doc_level_embedding)
                i += self.params.batch_size
            
            # for abstract_id in self.abstracts:
                
            #     sentence_level_embeddings = model.encode(self.abstracts[abstract_id], convert_to_numpy = True )
            #     doc_level_embedding = sentence_level_embeddings
            #     embeddings.append(doc_level_embedding)
            self.embeddings = np.array(embeddings)
            np.save(open(self.embeddings_file, "wb"), self.embeddings)

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

        self.X_train = X_train
        self.y_train = y_train

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        emb1 = self.embeddings[self.X_train[idx, 0]]
        emb2 = self.embeddings[self.X_train[idx, 1]]
        concatenated_embeddings = torch.cat((emb1, emb2), dim=0)
        label = self.y[idx]
        return concatenated_embeddings, label


