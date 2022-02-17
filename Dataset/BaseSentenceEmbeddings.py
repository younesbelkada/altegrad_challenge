import os
import os.path as osp
from random import randint

import networkx as nx
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
from utils.dataset_utils import get_abstracts_dict
from utils.logger import init_logger


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
        self.abstracts = get_abstracts_dict(path = path_txt)
        self.params = params
        self.embeddings_file = self.params.embeddings_file
        self.adj = None

    def get_features(self, edg0, edg1):

        list_features = [edg0, edg1]
        # Graph features
        # (1, 2) degree of two nodes
        # (3) sum of degrees of two nodes
        # (4) absolute value of difference of degrees of two nodes
        deg_edg0 = self.G.degree(edg0)
        deg_edg1 = self.G.degree(edg1)
        edg_sum = deg_edg0 + deg_edg1
        edg_abs = abs(deg_edg0 - deg_edg1)
        
        list_features.append(deg_edg0)
        list_features.append(deg_edg1)
        list_features.append(edg_sum)
        list_features.append(edg_abs)

        # Text features
        # (1) sum of number of unique terms of the two nodes' abstracts
        # (2) absolute value of difference of number of unique terms of the two nodes' abstracts
        # (3) number of common terms between the abstracts of the two nodes
        
        # TODO for each nodes detect keywords then intersection between two set of keywords 
        
        # Map text to set of terms
        set_1 = set(self.abstracts[edg0].split())
        set_2 = set(self.abstracts[edg1].split())

        sum_uni = len(set_1) + len(set_2)
        abs_uni = abs(len(set_1) - len(set_2))
        len_com_term = len(set_1.intersection(set_2))
        
        list_features.append(sum_uni)
        list_features.append(abs_uni)
        list_features.append(len_com_term)

        return tuple(list_features)


    def load_embeddings(self):
        '''
        https://huggingface.co/sentence-transformers/allenai-specter
        https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1
        https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1
        https://huggingface.co/allenai/scibert_scivocab_uncased
        '''
        
        if self.params.only_create_embeddings:
            self.logger.info("create embeddings enabled, creating the embeddings from scratch")
            
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

    def build_predict(self):
        
        self.logger.info("Building predict ...")

        self.predict_mode = True
        
        X = []
        with open(self.path_predict, 'r') as file:
            for line in file:
                line = line.split(',')
                edg0 = int(line[0])
                edg1 = int(line[1])
                X.append(self.get_features(edg0, edg1))

        
        X = np.array(X)
        X[:,2:] = (X[:,2:] - X[:,2:].min(0))/X[:,2:].ptp(0)
        self.X = X
        self.y = np.zeros(self.X.shape[0])

    def build_train(self):
        
        self.logger.info("Building train ...")

        self.predict_mode = False

        m = self.G.number_of_edges()
        n = self.G.number_of_nodes()
        X_train = np.zeros((2*m, 9))
        y_train = np.zeros(2*m)
        nodes = list(self.G.nodes())
        
        for i, edge in enumerate(self.G.edges()):

            X_train[2*i] = self.get_features(edge[0], edge[1])
            y_train[2*i] = 1 

            n1 = nodes[randint(0, n-1)] 
            n2 = nodes[randint(0, n-1)]
            # FIXME can create pairs of the test set
            while (n1, n2) in self.G.edges():
                n1 = nodes[randint(0, n-1)]
                n2 = nodes[randint(0, n-1)]

            X_train[2*i+1] = self.get_features(n1, n2)
            y_train[2*i+1] = 0

        self.logger.info("Finished building train ...")

        # normalize the last columns
        X_train[:,2:] = (X_train[:,2:] - X_train[:,2:].min(0))/X_train[:,2:].ptp(0)
        self.X = X_train
        self.y = y_train    

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        raise NotImplementedError(f'Should be implemented in derived class!')
