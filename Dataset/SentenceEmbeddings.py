from random import randint

import numpy as np
import torch
import random

from Dataset.BaseSentenceEmbeddings import BaseSentenceEmbeddings
import torch.nn.functional as F

from utils.dataset_utils import get_authors_dict

class SentenceEmbeddingsVanilla(BaseSentenceEmbeddings):
    def __init__(self, params) -> None:
        super().__init__(params, "SentenceEmbeddings")

    def build_predict(self):

        self.predict_mode = True
        
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
        emb1 = torch.from_numpy(self.abstract_embeddings[int(self.X[idx, 0])])
        emb2 = torch.from_numpy(self.abstract_embeddings[int(self.X[idx, 1])])
        
        concatenated_embeddings = torch.cat((emb1, emb2), dim=0)

        if not self.predict_mode:
            label = self.y[idx]
            return concatenated_embeddings, label
        return concatenated_embeddings


class SentenceEmbeddingsGraphAbstract(BaseSentenceEmbeddings):
    def __init__(self, params) -> None:
        super().__init__(params, "SentenceEmbeddingsGraphAbstract")

    def __getitem__(self, idx):

        # pos1 = random.randint(0, 1)
        # pos2 = 1-pos1
        pos1 = 0
        pos2 = 1

        emb1 = torch.from_numpy(self.abstract_embeddings[int(self.X[idx, pos1])])
        emb2 = torch.from_numpy(self.abstract_embeddings[int(self.X[idx, pos2])])

        # keyword_feat1 = torch.from_numpy(self.keywords_embeddings[pos1]).flatten()
        # keyword_feat2 = torch.from_numpy(self.keywords_embeddings[pos2]).flatten()

        keywords_emb1 = self.keywords_embeddings[pos1]
        keywords_emb2 = self.keywords_embeddings[pos2]

        keyword_feat_mean1 = torch.from_numpy(keywords_emb1).mean(dim=0).flatten()
        keyword_feat_mean2 = torch.from_numpy(keywords_emb2).mean(dim=0).flatten()

        cosine_feature = F.cosine_similarity(torch.from_numpy(keywords_emb1), torch.from_numpy(keywords_emb2), dim=1)
        pdist_feature = F.pdist(torch.from_numpy(keywords_emb1), torch.from_numpy(keywords_emb2), dim=1)
        concatenated_embeddings = torch.cat((cosine_feature, pdist_feature, emb1, emb2, keyword_feat_mean1, keyword_feat_mean2, torch.Tensor(self.X[idx, 2:])), dim=0)

        if not self.predict_mode:
            label = self.y[idx]
            return concatenated_embeddings, label
        return concatenated_embeddings


class SentenceEmbeddingsGraphAbstractTwo(BaseSentenceEmbeddings):
    def __init__(self, params) -> None:
        super().__init__(params, "SentenceEmbeddingsGraphAbstractTwo")

    def load_abstract_embeddings(self):
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
        
        concatenated_embeddings = torch.cat((emb1spec, emb2spec, emb1sci, emb2sci, torch.Tensor(self.X[idx, 2:])), dim=0)

        if not self.predict_mode:
            label = self.y[idx]
            return concatenated_embeddings, label
        return concatenated_embeddings


class SentenceEmbeddingsGraphWithNeighbors(BaseSentenceEmbeddings):
    def __init__(self, params) -> None:
        super().__init__(params, "SentenceEmbeddingsGraphWithNeighbors")

    def __getitem__(self, idx):
        current_node1, current_node2 = int(self.X[idx, 0]), int(self.X[idx, 1])

        emb1 = torch.from_numpy(self.abstract_embeddings[current_node1])
        emb2 = torch.from_numpy(self.abstract_embeddings[current_node2])

        neighbors_node1 = self.G.neighbors(current_node1)
        neighbors_node2 = self.G.neighbors(current_node2)

        neighbors_node1 = list(neighbors_node1)
        neighbors_node2 = list(neighbors_node2)
        
        mean_emb_n1 = torch.zeros(emb1.shape)
        mean = [torch.from_numpy(self.abstract_embeddings[n1]).unsqueeze(0) for n1 in neighbors_node1 if n1 != current_node2]
        if len(mean)>0:
            mean_emb_n1 = torch.mean(torch.cat(mean,dim=0), dim = 0)

        mean_emb_n2 = torch.zeros(emb2.shape)
        mean = [torch.from_numpy(self.abstract_embeddings[n2]).unsqueeze(0) for n2 in neighbors_node2 if n2 != current_node1]
        if len(mean)>0:
            mean_emb_n2 = torch.mean(torch.cat(mean,dim=0), dim = 0)

        concatenated_embeddings = torch.cat((emb1, emb2, mean_emb_n1, mean_emb_n2, torch.Tensor(self.X[idx][2:])), dim=0)

        if not self.predict_mode:
            label = self.y[idx]
            return concatenated_embeddings, label
        return concatenated_embeddings


class SentenceEmbeddingsFeatures(BaseSentenceEmbeddings):
    def __init__(self, params) -> None:
        super().__init__(params, "SentenceEmbeddingsFeatures")

    def __getitem__(self, idx):
        concatenated_embeddings = self.X[idx]
        if not self.predict_mode:
            label = self.y[idx]
            return concatenated_embeddings, label
        return concatenated_embeddings


class SentenceEmbeddingsGraphWithNeighborsAndAuthors(BaseSentenceEmbeddings):
    def __init__(self, params) -> None:
        super().__init__(params, "SentenceEmbeddingsGraphWithNeighborsAndAuthors")
        self.dict_authors, unique_authors = get_authors_dict(self.path_authors)
        self.unique_authors = len(unique_authors)
        self.dict_authors_to_index = {unique_authors[i]:i for i in range(len(unique_authors))}

        

    def __getitem__(self, idx):
        current_node1, current_node2 = int(self.X[idx, 0]), int(self.X[idx, 1])

        authors_node1, authors_node2 = self.dict_authors[current_node1][randint(0, len(self.dict_authors[current_node1])-1)], self.dict_authors[current_node2][randint(0, len(self.dict_authors[current_node2])-1)]
        authors_node1, authors_node2 = self.dict_authors_to_index[authors_node1], self.dict_authors_to_index[authors_node2]

        emb1 = torch.from_numpy(self.abstract_embeddings[current_node1])
        emb2 = torch.from_numpy(self.abstract_embeddings[current_node2])

        neighbors_node1 = self.G.neighbors(current_node1)
        neighbors_node2 = self.G.neighbors(current_node2)

        neighbors_node1 = list(neighbors_node1)
        neighbors_node2 = list(neighbors_node2)
        
        mean_emb_n1 = torch.zeros(emb1.shape)
        mean = [torch.from_numpy(self.abstract_embeddings[n1]).unsqueeze(0) for n1 in neighbors_node1 if n1 != current_node2]
        if len(mean)>0:
            mean_emb_n1 = torch.mean(torch.cat(mean,dim=0), dim = 0)

        mean_emb_n2 = torch.zeros(emb2.shape)
        mean = [torch.from_numpy(self.abstract_embeddings[n2]).unsqueeze(0) for n2 in neighbors_node2 if n2 != current_node1]
        if len(mean)>0:
            mean_emb_n2 = torch.mean(torch.cat(mean,dim=0), dim = 0)

        concatenated_embeddings = torch.cat((emb1, emb2, mean_emb_n1, mean_emb_n2, torch.Tensor(self.X[idx][2:])), dim=0)

        if not self.predict_mode:
            label = self.y[idx]
            return ((authors_node1, authors_node2), concatenated_embeddings), label
        return ((authors_node1, authors_node2), concatenated_embeddings)


class SentenceEmbeddingsGraphAbstractWithNeighbors(BaseSentenceEmbeddings):
    def __init__(self, params) -> None:
        super().__init__(params, "SentenceEmbeddingsGraphAbstractWithNeighbors")

    def __getitem__(self, idx):

        # pos1 = random.randint(0, 1)
        # pos2 = 1-pos1
        pos1 = 0
        pos2 = 1

        current_node1, current_node2 = int(self.X[idx, pos1]), int(self.X[idx, pos2])

        emb1 = torch.from_numpy(self.abstract_embeddings[current_node1])
        emb2 = torch.from_numpy(self.abstract_embeddings[current_node2])

        neighbors_node1 = list(self.G.neighbors(current_node1))
        neighbors_node2 = list(self.G.neighbors(current_node2))

        mean_emb_n1 = torch.zeros(emb1.shape)
        mean = [torch.from_numpy(self.abstract_embeddings[n1]).unsqueeze(0) for n1 in neighbors_node1 if n1 != current_node2]
        if len(mean)>0:
            mean_emb_n1 = torch.mean(torch.cat(mean,dim=0), dim = 0)

        mean_emb_n2 = torch.zeros(emb2.shape)
        mean = [torch.from_numpy(self.abstract_embeddings[n2]).unsqueeze(0) for n2 in neighbors_node2 if n2 != current_node1]
        if len(mean)>0:
            mean_emb_n2 = torch.mean(torch.cat(mean,dim=0), dim = 0)

        # keyword_feat1 = torch.from_numpy(self.keywords_embeddings[pos1]).flatten()
        # keyword_feat2 = torch.from_numpy(self.keywords_embeddings[pos2]).flatten()

        keyword_feat1 = torch.from_numpy(self.keywords_embeddings[pos1]).mean(dim=0).flatten()
        keyword_feat2 = torch.from_numpy(self.keywords_embeddings[pos2]).mean(dim=0).flatten()
        cosine_feature = F.cosine_similarity(torch.from_numpy(self.keywords_embeddings[pos1]), torch.from_numpy(self.keywords_embeddings[pos2]), dim=1)
        
        concatenated_embeddings = torch.cat((cosine_feature, emb1, emb2, keyword_feat1, keyword_feat2, mean_emb_n1, mean_emb_n2, torch.Tensor(self.X[idx, 2:])), dim=0)

        if not self.predict_mode:
            label = self.y[idx]
            return concatenated_embeddings, label
        return concatenated_embeddings