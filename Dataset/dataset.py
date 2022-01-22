from random import randint
import torch
import networkx as nx
import numpy as np
import logging
import os.path as osp

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

