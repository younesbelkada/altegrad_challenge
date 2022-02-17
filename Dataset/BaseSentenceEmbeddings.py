import os
import os.path as osp
from random import randint

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
from utils.dataset_utils import get_abstracts_dict
from utils.logger import init_logger
from keybert import KeyBERT

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

        self.abstract_embeddings_file = self.params.abstract_embeddings_file
        self.keywords_embeddings_file = self.params.keywords_embeddings_file
        self.keywords_file = self.params.keywords_file
        self.nb_keywords = self.params.nb_keywords

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

        # Map text to set of terms
        # set_1 = set(self.abstracts[edg0].split()) # slow
        # set_2 = set(self.abstracts[edg1].split()) # slow

        # sum_uni = len(set_1) + len(set_2)
        # abs_uni = abs(len(set_1) - len(set_2))
        # len_com_term = len(set_1.intersection(set_2))
        
        # list_features.append(sum_uni)
        # list_features.append(abs_uni)
        # list_features.append(len_com_term)
        
        # (4, 5) keywords length and intersection
        # set_key_1 = set(self.keywords[edg0]) 
        # set_key_2 = set(self.keywords[edg1]) 

        # len_keywords = len(set_key_1)
        # len_com_keywords = len(set_key_1.intersection(set_key_2))
        
        # list_features.append(len_keywords)
        # list_features.append(len_com_keywords)
        
        # Keywords features 

        # keyword_feat1 = self.keywords_embeddings[edg0]
        # keyword_feat2 = self.keywords_embeddings[edg1]

        # list_features.append(keyword_feat1)
        # list_features.append(keyword_feat2)

        return tuple(list_features)

    def load_keywords(self):

        if self.params.only_create_keywords:
            self.logger.info("create keywords enabled, creating the keywords from scratch")

            self.logger.info(self.params.name_transformer)

            keywords_array = []
            keywords_embed = []

            self.logger.info(f"batch size: {self.params.batch_size}")

            i = 0
            model = SentenceTransformer(self.params.name_transformer).to(self.device)
            kw_model = KeyBERT(model=model)
            
            while i < len(self.abstracts):
                abstracts_batch = [self.abstracts[abstract_id] for abstract_id in list(self.abstracts.keys())[i:min(i+self.params.batch_size, len(self.abstracts))]]
                keywords_batch = kw_model.extract_keywords(abstracts_batch, 
                                                    keyphrase_ngram_range=(1, 2),
                                                    use_maxsum=True, 
                                                    top_n=self.nb_keywords,
                                                    stop_words='english')

                converted_keywords_batch = [[b[0] for b in batch] if len(batch) == self.nb_keywords else ['N' for _ in range(self.nb_keywords)] for batch in keywords_batch]
                converted_keywords_batch = np.array(converted_keywords_batch).flatten()
                keywords_embeddings = model.encode(converted_keywords_batch, convert_to_numpy = True, show_progress_bar=False)
                keywords_embed.extend(keywords_embeddings.reshape(len(keywords_batch), self.nb_keywords, keywords_embeddings.shape[1]))
                keywords_array.extend(keywords_batch)
                i += self.params.batch_size

            self.keywords = np.array(keywords_array)
            self.keywords_embed = np.array(keywords_embed)
            raw_keywords_name_file = osp.join(os.getcwd(), "input", f"keywords-{self.nb_keywords}-{self.params.name_transformer.replace('/', '_')}.npy")
            emb_keywords_name_file = osp.join(os.getcwd(), "input", f"keywords_emb-{self.nb_keywords}-{self.params.name_transformer.replace('/', '_')}.npy")
            
            np.save(open(raw_keywords_name_file, "wb"), self.keywords)
            self.logger.info(f"{raw_keywords_name_file} created")
            
            np.save(open(emb_keywords_name_file, "wb"), self.keywords_embed)
            self.logger.info(f"{emb_keywords_name_file} created")

        elif os.path.isfile(self.keywords_file) and os.path.isfile(self.keywords_embeddings_file):
            
            self.logger.info("Keywords file already exists, loading it directly")
            self.logger.info(f"Loading {self.keywords_file}")
            self.keywords = np.load(open(self.keywords_file, 'rb'), allow_pickle=True)
            
            self.logger.info("Keywords embeddings file already exists, loading it directly")
            self.logger.info(f"Loading {self.keywords_embeddings_file}")
            self.keywords_embeddings = np.load(open(self.keywords_embeddings_file, 'rb'))

    def load_abstract_embeddings(self):
        '''
        https://huggingface.co/sentence-transformers/allenai-specter
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
                abstract_level_embeddings = model.encode(abstracts_batch, convert_to_numpy = True, show_progress_bar=False)
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
        self.X = X
        self.y = np.zeros(self.X.shape[0])

    def build_train(self):
        
        self.logger.info("Building train ...")

        self.predict_mode = False

        m = self.G.number_of_edges()
        n = self.G.number_of_nodes()
        X_train = np.zeros((2*m, 6))
        y_train = np.zeros(2*m)

        nodes = list(self.G.nodes())
        
        for i, edge in enumerate(self.G.edges()):

            X_train[2*i] = self.get_features(edge[0], edge[1])
            y_train[2*i] = 1 

            n1 = nodes[randint(0, n-1)] 
            n2 = nodes[randint(0, n-1)]

            # FIXME can create pairs of the test set try without negative pairs
            while (n1, n2) in self.G.edges():
                n1 = nodes[randint(0, n-1)]
                n2 = nodes[randint(0, n-1)]

            X_train[2*i+1] = self.get_features(n1, n2)
            y_train[2*i+1] = 0

        self.logger.info("Finished building train ...")

        self.X = X_train
        self.y = y_train    

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        raise NotImplementedError(f'Should be implemented in derived class!')
