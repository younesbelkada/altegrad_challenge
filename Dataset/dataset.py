import logging
import os
import os.path as osp
from random import randint

import networkx as nx
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
#from torch_geometric.utils.convert import from_scipy_sparse_matrix
from utils.dataset_utils import get_abstracts_dict
from utils.logger import init_logger


class BaselineGraphDataset(object):
    def __init__(self, params):
        """
            :- params -: Parameters object
        """
        logging.info("Loading the Graph object from the edgelist")
        path_edges = osp.join(params.root_dataset, 'edgelist.txt')
        self.G = nx.read_edgelist(
            path_edges, delimiter=',', create_using=nx.Graph(), nodetype=int)

    def build_train(self):
        logging.info("Building the training dataset using the nx graph")
        nodes = list(self.G.nodes())
        n = self.G.number_of_nodes()
        m = self.G.number_of_edges()

        X_train = np.zeros((2*m, 2))
        y_train = np.zeros(2*m)
        for i, edge in enumerate(self.G.edges()):
            # an edge
            X_train[2*i, 0] = self.G.degree(edge[0]) + self.G.degree(edge[1])
            X_train[2*i, 1] = abs(self.G.degree(edge[0]) -
                                  self.G.degree(edge[1]))
            y_train[2*i] = 1

            # a randomly generated pair of nodes
            n1 = nodes[randint(0, n-1)]
            n2 = nodes[randint(0, n-1)]
            X_train[2*i+1, 0] = self.G.degree(n1) + self.G.degree(n2)
            X_train[2*i+1, 1] = abs(self.G.degree(n1) - self.G.degree(n2))
            y_train[2*i+1] = 0

        logging.info('Size of training matrix: {}'.format(X_train.shape))
        self.X = X_train
        self.y = y_train

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(np.array(self.y[idx]))
        output_dict = {'x': x, 'y': y}
        return output_dict

    def __len__(self):
        return self.y.shape[0]


class GraphAutoEncoderDataset(Dataset):
    def __init__(self, params) -> None:
        super().__init__()

        self.logger = init_logger("GraphAutoEncoderDataset", "INFO")

        self.logger.info(
            "Loading the Graph object from the edgelist and the embeddings obtained using the abstracts...")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.Tensor(self.X[idx]).long()
        return x, self.adj


# class SubGraphsDataset(Dataset):
#     def __init__(self, params) -> None:
#         super().__init__()
#         self.params = params

#         self.logger = init_logger("SubGraphsDataset", "INFO")

#         self.logger.info(
#             "Loading the Graph object from the edgelist and the embeddings obtained using the abstracts...")
#         path_txt = osp.join(params.root_dataset, 'abstracts.txt')
#         path_edges = osp.join(params.root_dataset, 'edgelist.txt')
#         self.path_predict = osp.join(params.root_dataset, 'test.txt')
#         self.G = nx.read_edgelist(
#             path_edges, delimiter=',', create_using=nx.Graph(), nodetype=int)
#         self.device = torch.device(
#             "cuda") if torch.cuda.is_available() else torch.device("cpu")

#         self.nodes = list(self.G.nodes())
#         m = self.G.number_of_edges()
#         n = self.G.number_of_nodes()
#         X_train = np.zeros((2*m, 2))
#         y_train = np.zeros(2*m)
#         for i, edge in enumerate(self.G.edges()):

#             X_train[2*i] = edge
#             y_train[2*i] = 1

#             n1 = self.nodes[randint(0, n-1)]
#             n2 = self.nodes[randint(0, n-1)]

#             X_train[2*i+1] = (n1, n2)
#             y_train[2*i+1] = 0
#         self.X = X_train
#         self.y = y_train

#     def build_train(self):
#         self.predict_mode = False
#         model = SentenceTransformer(
#             'sentence-transformers/allenai-specter').to(self.device)
#         embeddings = []
#         labels = []

#         if self.params.force_create:
#             self.logger.info(
#                 "Force create enabled, creating the embeddings from scratch")
#             i = 0
#             while i < len(self.abstracts):
#                 abstracts_batch = [self.abstracts[abstract_id] for abstract_id in list(
#                     self.abstracts.keys())[i:min(i+self.params.batch_size, len(self.abstracts))]]
#                 sentence_level_embeddings = model.encode(
#                     abstracts_batch, convert_to_numpy=True, show_progress_bar=False)
#                 doc_level_embedding = sentence_level_embeddings
#                 embeddings.extend(doc_level_embedding)
#                 i += self.params.batch_size

#             self.embeddings = np.array(embeddings)
#             np.save(open(self.embeddings_file, "wb"), self.embeddings)
#         elif os.path.isfile(self.embeddings_file):
#             self.logger.info(
#                 "Embedings file already exists, loading it directly")
#             self.logger.info(f"Loading {self.embeddings_file}")
#             self.embeddings = np.load(open(self.embeddings_file, 'rb'))

#         # Comment créer les labels?
#         m = self.G.number_of_edges()
#         n = self.G.number_of_nodes()
#         X_train = np.zeros((2*m, 2))
#         y_train = np.zeros(2*m)
#         nodes = list(self.G.nodes())
#         for i, edge in enumerate(self.G.edges()):

#             X_train[2*i] = edge
#             y_train[2*i] = 1

#             n1 = nodes[randint(0, n-1)]
#             n2 = nodes[randint(0, n-1)]

#             X_train[2*i+1] = (n1, n2)
#             y_train[2*i+1] = 0

#         self.X = X_train
#         self.y = y_train

#     def build_predict(self):
#         self.predict_mode = True
#         if os.path.isfile(self.embeddings_file):
#             self.logger.info(
#                 "Embedings file already exists, loading it directly")
#             self.logger.info(f"Loading {self.embeddings_file}")
#             self.embeddings = np.load(open(self.embeddings_file, 'rb'))
#         else:
#             raise NotImplementedError

#         X = []
#         with open(self.path_predict, 'r') as file:
#             for line in file:
#                 line = line.split(',')
#                 X.append((int(line[0]), int(line[1])))
#         self.X = np.array(X)
#         self.y = np.zeros(self.X.shape[0])

#     def __getitem__(self, idx):
#         current_node1, current_node2 = self.X[idx, 0], self.X[idx, 1]

#         #subgraph_nodes1 = self.subgraphs[current_node1]
#         #subgraph_nodes2 = self.subgraphs[current_node2]
#         neighbors_node1 = self.G.neighbors(current_node1)
#         neighbors_node2 = self.G.neighbors(current_node2)

#         neighbors_node1 = list(neighbors_node1)
#         neighbors_node2 = list(neighbors_node2)

#         label = self.y[idx]

#         output_dict = {
#             "neighbors_node1": neighbors_node1,
#             "neighbors_node2": neighbors_node2,
#             "label": label
#         }

#         neighbors_node1.insert(0, current_node1)
#         neighbors_node2.insert(0, current_node2)

#         subgraph1 = self.G.subgraph(neighbors_node1)
#         subgraph2 = self.G.subgraph(neighbors_node2)

#         output_dict["adj1"] = nx.adjacency_matrix(subgraph1)
#         output_dict["adj2"] = nx.adjacency_matrix(subgraph2)

#         return output_dict

#     def __len__(self):
#         return self.y.shape[0]

#     def collate_fn(self, batch):
#         neighbors_node1 = pad_sequence([torch.LongTensor(
#             sample['neighbors_node1']) for sample in batch], padding_value=self.params.vocab_size)
#         neighbors_node2 = pad_sequence([torch.LongTensor(
#             sample['neighbors_node2']) for sample in batch], padding_value=self.params.vocab_size)

#         max_len1 = max([sample['adj1'].shape[0] for sample in batch])
#         max_len2 = max([sample['adj2'].shape[0] for sample in batch])

#         #adj1 = F.pad(([sample['adj1'] for sample in batch]), value=-1)
#         adj1 = [from_scipy_sparse_matrix(sample['adj1'])[0]
#                 for sample in batch]
#         #adj2 = F.pad([torch.LongTensor(sample['adj2']) for sample in batch], value=-1)
#         adj2 = [from_scipy_sparse_matrix(sample['adj2'])[0]
#                 for sample in batch]

#         labels = [int(sample['label']) for sample in batch]

#         neighbors_node1 = neighbors_node1.transpose(1, 0)
#         neighbors_node2 = neighbors_node2.transpose(1, 0)

#         return (neighbors_node1, neighbors_node2), torch.Tensor(labels)


class SubGraphsDatasetWithoutAdj(Dataset):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params

        self.logger = init_logger("SubGraphsDataset", "INFO")

        self.logger.info(
            "Loading the Graph object from the edgelist and the embeddings obtained using the abstracts...")
        path_txt = osp.join(params.root_dataset, 'abstracts.txt')
        path_edges = osp.join(params.root_dataset, 'edgelist.txt')
        self.path_predict = osp.join(params.root_dataset, 'test.txt')
        self.G = nx.read_edgelist(
            path_edges, delimiter=',', create_using=nx.Graph(), nodetype=int)
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.abstracts = get_abstracts_dict(path=path_txt)
        self.params = params
        self.embeddings_file = self.params.embeddings_file
        self.adj = None

        self.nodes = list(self.G.nodes())
        m = self.G.number_of_edges()
        n = self.G.number_of_nodes()
        X_train = np.zeros((2*m, 2))
        y_train = np.zeros(2*m)
        for i, edge in enumerate(self.G.edges()):

            X_train[2*i] = edge
            y_train[2*i] = 1

            n1 = self.nodes[randint(0, n-1)]
            n2 = self.nodes[randint(0, n-1)]

            X_train[2*i+1] = (n1, n2)
            y_train[2*i+1] = 0
        self.X = X_train
        self.y = y_train

    def build_train(self):
        pass

    def build_predict(self):
        pass

    def __getitem__(self, idx):
        current_node1, current_node2 = self.X[idx, 0], self.X[idx, 1]

        #subgraph_nodes1 = self.subgraphs[current_node1]
        #subgraph_nodes2 = self.subgraphs[current_node2]
        neighbors_node1 = self.G.neighbors(current_node1)
        neighbors_node2 = self.G.neighbors(current_node2)

        neighbors_node1 = list(neighbors_node1)
        neighbors_node2 = list(neighbors_node2)

        label = self.y[idx]

        output_dict = {
            "neighbors_node1": neighbors_node1,
            "neighbors_node2": neighbors_node2,
            "label": label
        }

        neighbors_node1.insert(0, current_node1)
        neighbors_node2.insert(0, current_node2)

        neighbors_emb1 = []
        neighbors_emb2 = []
        for node1 in neighbors_node1:
            neighbors_emb1.append(torch.from_numpy(
                self.embeddings[int(self.X[node1, 0])]))
        for node2 in neighbors_node2:
            neighbors_emb2.append(torch.from_numpy(
                self.embeddings[int(self.X[node2, 1])]))

        output_dict["emb1"] = neighbors_emb1
        output_dict["emb2"] = neighbors_emb2

        return output_dict

    def __len__(self):
        return self.y.shape[0]

    def collate_fn(self, batch):
        neighbors_node1 = pad_sequence([torch.LongTensor(
            sample['neighbors_node1']) for sample in batch], padding_value=self.params.vocab_size)
        neighbors_node2 = pad_sequence([torch.LongTensor(
            sample['neighbors_node2']) for sample in batch], padding_value=self.params.vocab_size)
        labels = [int(sample['label']) for sample in batch]

        neighbors_node1 = neighbors_node1.transpose(1, 0)
        neighbors_node2 = neighbors_node2.transpose(1, 0)

        return (neighbors_node1, neighbors_node2), torch.Tensor(labels)
