import os
import os.path as osp
from random import randint

import networkx as nx
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
from utils.dataset_utils import get_specter_abstracts_dict
from utils.logger import init_logger
from wordwise import Extractor
from summa import keywords

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

        self.abstract_embeddings_file = self.params.abstract_embeddings_file
        self.keywords_embeddings_file = self.params.keywords_embeddings_file
        self.keywords_file = self.params.keywords_file
        self.nb_keywords = self.params.nb_keywords

    def get_features(self, edg0, edg1):
        # Graph features
        # (1, 2) degree of two nodes
        # (3) sum of degrees of two nodes
        # (4) absolute value of difference of degrees of two nodes
        deg_edg0 = self.G.degree(edg0)
        deg_edg1 = self.G.degree(edg1)
        edg_sum = deg_edg0 + deg_edg1
        edg_abs = abs(deg_edg0 - deg_edg1)
        
        # Text features
        # (1) sum of number of unique terms of the two nodes' abstracts
        # (2) absolute value of difference of number of unique terms of the two nodes' abstracts
        # (3) number of common terms between the abstracts of the two nodes

        # Map text to set of terms
        set_1 = set(self.abstracts[edg0].split())
        set_2 = set(self.abstracts[edg1].split())

        sum_uni = len(set_1) + len(set_2)
        abs_uni = abs(len(set_1) - len(set_2))
        len_com_term = len(set_1.intersection(set_2))

        return (edg0, edg1, deg_edg0, deg_edg1, edg_sum, edg_abs, sum_uni, abs_uni, len_com_term)

    def get_features(self, edg0, edg1):
        # Graph features
        # (1, 2) degree of two nodes
        # (3) sum of degrees of two nodes
        # (4) absolute value of difference of degrees of two nodes
        deg_edg0 = self.G.degree(edg0)
        deg_edg1 = self.G.degree(edg1)
        edg_sum = deg_edg0 + deg_edg1
        edg_abs = abs(deg_edg0 - deg_edg1)
        
        # Text features
        # (1) sum of number of unique terms of the two nodes' abstracts
        # (2) absolute value of difference of number of unique terms of the two nodes' abstracts
        # (3) number of common terms between the abstracts of the two nodes

        # Map text to set of terms
        set_1 = set(self.abstracts[edg0].split())
        set_2 = set(self.abstracts[edg1].split())

        sum_uni = len(set_1) + len(set_2)
        abs_uni = abs(len(set_1) - len(set_2))
        len_com_term = len(set_1.intersection(set_2))

        # (4, 5) keywords length and intersection
        set_key_1 = set(self.keywords[edg0])
        set_key_2 = set(self.keywords[edg1])

        len_keywords = len(set_key_1)
        len_com_keywords = len(set_key_1.intersection(set_key_2))
        
        return (edg0, edg1, deg_edg0, deg_edg1, edg_sum, edg_abs, sum_uni, abs_uni, len_com_term, len_com_keywords, len_keywords)

    def load_keywords(self):

        if self.params.only_create_keywords:
            self.logger.info("create keywords enabled, creating the keywords from scratch")

            self.logger.info(self.params.name_transformer)

            keywords = []

            # method 1: leverage BERT features https://jaketae.github.io/study/keyword-extraction/
            # https://github.com/jaketae/wordwise/blob/master/wordwise/core.py

            extractor = Extractor(        
                n_gram_range=(1, 1),
                spacy_model="en_core_web_sm",
                # bert_model="sentence-transformers/all-MiniLM-L12-v2", # by default (change?)
                bert_model=self.params.name_transformer,
                device=self.device
                )

            for abstract_id in list(self.abstracts.keys()):
                keywords.append(extractor.generate(self.abstracts[abstract_id], self.nb_keywords))

            self.keywords = np.array(keywords)
            name_file = osp.join(os.getcwd(), "input", f"keywords-{self.nb_keywords}-{self.params.name_transformer.replace('/', '_')}.npy")
            np.save(open(name_file, "wb"), self.keywords)
            
            # method 2: Text Rank https://medium.com/mlearning-ai/10-popular-keyword-extraction-algorithms-in-natural-language-processing-8975ada5750c
            # https://github.com/summanlp/textrank
            
            # for abstract_id in list(self.abstracts.keys()):
            #     keywords.append(keywords.keywords(self.abstracts[abstract_id], words=self.nb_keywords))
            
            # self.keywords = np.array(keywords)
            # name_file = osp.join(os.getcwd(), "input", f"keywords-{self.nb_keywords}-textrank.npy")
            # np.save(open(name_file, "wb"), self.keywords)
            
        elif os.path.isfile(self.keywords_file):
            self.logger.info("Keywords file already exists, loading it directly")
            self.logger.info(f"Loading {self.keywords_file}")
            self.keywords = np.load(open(self.keywords_file, 'rb'))

    def load_keywords_embeddings(self):
        '''
        # TODO
        '''
        
        # if self.params.only_create_keywords_embeddings:
        #     self.logger.info("create keywords embeddings enabled, creating the keywords embeddings from scratch")
            
        #     self.logger.info(self.params.name_transformer)
        #     model = SentenceTransformer(self.params.name_transformer).to(self.device)
        #     keywords_embeddings = []
            
        #     self.logger.info(f"batch size: {self.params.batch_size}")

        #     i = 0
        #     while i < len(self.abstracts):
        #         abstracts_batch = [self.abstracts[abstract_id] for abstract_id in list(self.abstracts.keys())[i:min(i+self.params.batch_size, len(self.abstracts))]]
        #         sentence_level_embeddings = model.encode(abstracts_batch, convert_to_numpy = True, show_progress_bar=False)
        #         keywords_embeddings.extend(doc_level_embedding)
        #         i += self.params.batch_size
            
        #     self.keywords_embeddings = np.array(keywords_embeddings)
        #     name_file = osp.join(os.getcwd(), "input", f"keywords_embeddings-{self.nb_keywords}-{self.params.name_transformer.replace('/', '_')}.npy")
        #     np.save(open(name_file, "wb"), self.keywords_embeddings)
        
        # elif os.path.isfile(self.keywords_embeddings_file):
        #     self.logger.info("Embedings file already exists, loading it directly")
        #     self.logger.info(f"Loading {self.keywords_embeddings_file}")
        #     self.keywords_embeddings = np.load(open(self.keywords_embeddings_file, 'rb'))

    def load_abstract_embeddings(self):
        '''
        https://huggingface.co/sentence-transformers/allenai-specter
        https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1
        https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1
        https://huggingface.co/allenai/scibert_scivocab_uncased
        '''
        
        if self.params.only_create_abstract_embeddings:
            self.logger.info("create abstract embeddings enabled, creating the abstract embeddings from scratch")
            
            self.logger.info(self.params.name_transformer)
            model = SentenceTransformer(self.params.name_transformer).to(self.device)
            abstract_embeddings = []
            
            self.logger.info(f"batch size: {self.params.batch_size}")

            i = 0
            while i < len(self.abstracts):
                abstracts_batch = [self.abstracts[abstract_id] for abstract_id in list(self.abstracts.keys())[i:min(i+self.params.batch_size, len(self.abstracts))]]
                abstract_level_embeddings = model.encode(abstracts_batch, convert_to_numpy = True, show_progress_bar=False) # FIXME encode que par rapport aux abstract du batch ? 
                abstract_embeddings.extend(abstract_level_embeddings)
                i += self.params.batch_size
            
            self.abstract_embeddings = np.array(abstract_embeddings)
            name_file = osp.join(os.getcwd(), "input", f"abstract_embeddings-{self.params.name_transformer.replace('/', '_')}.npy")
            np.save(open(name_file, "wb"), self.abstract_embeddings)
        
        elif os.path.isfile(self.abstract_embeddings_file):
            self.logger.info("Embedings file already exists, loading it directly")
            self.logger.info(f"Loading {self.abstract_embeddings_file}")
            self.abstract_embeddings = np.load(open(self.abstract_embeddings_file, 'rb'))

    def build_predict(self):
        
        self.predict_mode = True

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

            while (n1,n2) in  self.G.edges():
                n1 = nodes[randint(0, n-1)]
                n2 = nodes[randint(0, n-1)]

            X_train[2*i+1] = self.get_features(n1, n2)
            y_train[2*i+1] = 0

        self.X = X_train
        self.y = y_train    

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        raise NotImplementedError(f'Should be implemented in derived class!')
