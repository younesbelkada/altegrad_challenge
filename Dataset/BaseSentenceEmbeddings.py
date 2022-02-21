import os
import os.path as osp
from random import randint

import networkx as nx
import numpy as np
import torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
from utils.dataset_utils import (get_abstracts_dict, get_authors_dict,
                                 get_progress_bar)
from utils.logger import init_logger


class BaseSentenceEmbeddings(Dataset):
    def __init__(self, params, name_dataset) -> None:
        super().__init__()

        self.logger = init_logger(name_dataset, "INFO")

        self.logger.info(
            "Loading the Graph object from the edgelist and the embeddings obtained using the abstracts...")
        path_txt = osp.join(params.root_dataset, 'abstracts.txt')
        path_edges = osp.join(params.root_dataset, 'edgelist.txt')
        self.path_predict = osp.join(params.root_dataset, 'test.txt')
        self.path_authors = osp.join(params.root_dataset, 'authors.txt')
        self.G = nx.read_edgelist(
            path_edges, delimiter=',', create_using=nx.Graph(), nodetype=int)

        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.params = params

        self.embed_param = params.embed_param

        if self.embed_param.use_keywords_embeddings or self.embed_param.use_abstract_embeddings:
            self.abstracts = get_abstracts_dict(path=path_txt)

        if self.embed_param.use_abstract_embeddings or self.embed_param.use_neighbors_embeddings:
            self.abstract_embeddings_file = self.params.abstract_embeddings_file
            self.load_abstract_embeddings()

        if self.embed_param.use_keywords_embeddings:
            self.keywords_embeddings_file = self.params.keywords_embeddings_file
            self.keywords_file = self.params.keywords_file
            self.nb_keywords = self.params.nb_keywords
            self.load_keywords()

        if self.embed_param.use_authors_embeddings:
            self.dict_authors, unique_authors = get_authors_dict(
                self.path_authors)
            self.unique_authors = len(unique_authors)
            self.dict_authors_to_index = {
                unique_authors[i]: i for i in range(len(unique_authors))}

    def get_input_dim(self):
        return self.get_final_embeddings(0, 1).shape[0]

    def get_handcrafted_embeddings(self, edg0, edg1):

        list_features = []
        # Graph features
        # (1, 2) degree of two nodes
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

        return torch.tensor(list_features)

    def get_jaccard_coefficient(self, current_node1, current_node2):
        jac = nx.jaccard_coefficient(self.G, [(current_node1, current_node2)])
        return [list(jac)[-1][-1]]

    # def get_eigenvector_centrality(self,e1,e2):
    #     return nx.eigenvector_centrality(self.G)

    def get_clustering(self, e1, e2):
        res = list(nx.clustering(self.G, [e1, e2]).values())
        if len(res) != 2:
            res = [0, 0]
        return res

    def get_adamic_adar_index(self, e1, e2):
        res = []
        preds = nx.adamic_adar_index(self.G, [(e1, e2)])
        try:
            for u, v, p in preds:
                res.append(p)
            return res
        except:
            return [0]

    def get_preferential_attachment(self, e1, e2):
        res = []
        preds = nx.preferential_attachment(self.G, [(e1, e2)])
        try:
            for u, v, p in preds:
                res.append(p)
            return res
        except:
            return [0]

    def get_cn_soundarajan_hopcroft(self, e1, e2):
        res = []
        preds = nx.cn_soundarajan_hopcroft(self.G, [(e1, e2)])
        try:
            for u, v, p in preds:
                res.append(p)
            return res
        except:
            return [0]

    def get_ra_index_soundarajan_hopcroft(self, e1, e2):
        res = []
        preds = nx.ra_index_soundarajan_hopcroft(self.G, [(e1, e2)])
        try:
            for u, v, p in preds:
                res.append(p)
            return res
        except:
            return [0]

    def get_shortest_path(self, e1, e2):
        return [nx.shortest_path_length(self.G, e1, e2)]

    def get_common_neighbor_centrality(self, e1, e2):
        res = []
        preds = nx.common_neighbor_centrality(self.G, (e1, e2))
        try:
            for u, v, p in preds:
                res.append(p)
            return res
        except:
            return [0]

    def get_sorenson_index(self, e1, e2):

        deg_edg0 = self.G.degree(e1)
        deg_edg1 = self.G.degree(e2)

        neighbors_node1 = self.G.neighbors(e1)
        neighbors_node2 = self.G.neighbors(e2)

        neighbors_node1 = list(neighbors_node1)
        neighbors_node2 = list(neighbors_node2)

        intersection = len(set(neighbors_node1).intersection(neighbors_node2))

        if (deg_edg0+deg_edg1) == 0:
            return [0]
        return [(2*intersection)/(deg_edg0+deg_edg1)]

    def get_neighbors_embeddings(self, current_node1, current_node2, emb_dim=768):

        neighbors_node1 = self.G.neighbors(current_node1)
        neighbors_node2 = self.G.neighbors(current_node2)

        neighbors_node1 = list(neighbors_node1)
        neighbors_node2 = list(neighbors_node2)

        intersection = len(set(neighbors_node1).intersection(neighbors_node2))

        mean_emb_n1 = torch.zeros(emb_dim)
        mean = [torch.from_numpy(self.abstract_embeddings[n1]).unsqueeze(
            0) for n1 in neighbors_node1 if n1 != current_node2]
        if len(mean) > 0:
            mean_emb_n1 = torch.mean(torch.cat(mean, dim=0), dim=0)

        mean_emb_n2 = torch.zeros(emb_dim)
        mean = [torch.from_numpy(self.abstract_embeddings[n2]).unsqueeze(
            0) for n2 in neighbors_node2 if n2 != current_node1]
        if len(mean) > 0:
            mean_emb_n2 = torch.mean(torch.cat(mean, dim=0), dim=0)

        return torch.cat((mean_emb_n1, mean_emb_n2, torch.Tensor([intersection])))

    def get_abstract_embeddings(self, current_node1, current_node2):

        emb1 = torch.from_numpy(self.abstract_embeddings[current_node1])
        emb2 = torch.from_numpy(self.abstract_embeddings[current_node2])

        return torch.cat((emb1, emb2))

    def get_keywords_embeddings(self, current_node1, current_node2):

        # keywords_emb1 = torch.from_numpy(self.keywords_embeddings[current_node1])
        # keywords_emb2 = torch.from_numpy(self.keywords_embeddings[current_node2])

        # keyword_feat_mean1 = keywords_emb1.mean(dim=0)
        # keyword_feat_mean2 = keywords_emb2.mean(dim=0)

        # cosine_feature = F.cosine_similarity(keywords_emb1, keywords_emb2, dim=1)
        # pdist_feature =  F.pdist(torch.cat((keywords_emb1, keywords_emb2)))

        # (4, 5) keywords length and intersection
        set_key_1 = set([k[0] if isinstance(k, tuple)
                        else k for k in self.keywords[current_node1]])
        set_key_2 = set([k[0] if isinstance(k, tuple)
                        else k for k in self.keywords[current_node2]])

        if 'N' in set_key_1 or 'N' in set_key_2:
            len_keywords = 0
            len_com_keywords = 0
        else:
            len_keywords = len(set_key_1)
            len_com_keywords = len(set_key_1.intersection(set_key_2))

        # return torch.cat((keyword_feat_mean1, keyword_feat_mean2, cosine_feature, pdist_feature))
        # return torch.cat((cosine_feature, pdist_feature))
        return torch.tensor([len_com_keywords])

    def get_authors_embeddings(self, current_node1, current_node2):

        authors_node1, authors_node2 = self.dict_authors[current_node1][randint(0, len(
            self.dict_authors[current_node1])-1)], self.dict_authors[current_node2][randint(0, len(self.dict_authors[current_node2])-1)]
        authors_node1, authors_node2 = self.dict_authors_to_index[
            authors_node1], self.dict_authors_to_index[authors_node2]

    def get_final_embeddings(self, node1, node2):
        final_embeddings = torch.tensor([])
        params = (self.embed_param).__dict__

        # remove their connexion as if we were testing
        for i, embed in enumerate(params):
            if params[embed]:
                fct = getattr(self, embed.replace("use", "get"))
                nodes_embeddings = fct(node1, node2)
                if not torch.is_tensor(nodes_embeddings):
                    nodes_embeddings = torch.tensor(nodes_embeddings)
                final_embeddings = torch.cat(
                    (final_embeddings, nodes_embeddings))

        return final_embeddings

    def load_keywords(self):

        if self.params.only_create_keywords:
            self.logger.info(
                "create keywords enabled, creating the keywords from scratch")

            self.logger.info(self.params.name_transformer)

            keywords_array = []
            keywords_embed = []

            self.logger.info(f"batch size: {self.params.batch_size}")

            i = 0
            model = SentenceTransformer(
                self.params.name_transformer).to(self.device)
            kw_model = KeyBERT(model=model)

            while i < len(self.abstracts):
                abstracts_batch = [self.abstracts[abstract_id] for abstract_id in list(
                    self.abstracts.keys())[i:min(i+self.params.batch_size, len(self.abstracts))]]
                keywords_batch = kw_model.extract_keywords(abstracts_batch,
                                                           keyphrase_ngram_range=(
                                                               1, 2),
                                                           use_maxsum=True,
                                                           top_n=self.nb_keywords,
                                                           stop_words='english')

                converted_keywords_batch = [[b[0] for b in batch] if len(batch) == self.nb_keywords else [
                    'N' for _ in range(self.nb_keywords)] for batch in keywords_batch]
                converted_keywords_batch = np.array(
                    converted_keywords_batch).flatten()
                keywords_embeddings = model.encode(
                    converted_keywords_batch, convert_to_numpy=True, show_progress_bar=False)
                keywords_embed.extend(keywords_embeddings.reshape(
                    len(keywords_batch), self.nb_keywords, keywords_embeddings.shape[1]))
                keywords_array.extend(keywords_batch)
                i += self.params.batch_size

            self.keywords = np.array(keywords_array)
            self.keywords_embed = np.array(keywords_embed)
            raw_keywords_name_file = osp.join(os.getcwd(
            ), "input", f"keywords-{self.nb_keywords}-{self.params.name_transformer.replace('/', '_')}.npy")
            emb_keywords_name_file = osp.join(os.getcwd(
            ), "input", f"keywords_emb-{self.nb_keywords}-{self.params.name_transformer.replace('/', '_')}.npy")

            np.save(open(raw_keywords_name_file, "wb"), self.keywords)
            self.logger.info(f"{raw_keywords_name_file} created")

            np.save(open(emb_keywords_name_file, "wb"), self.keywords_embed)
            self.logger.info(f"{emb_keywords_name_file} created")

        elif os.path.isfile(self.keywords_file) and os.path.isfile(self.keywords_embeddings_file):

            self.logger.info(
                "Keywords file already exists, loading it directly")
            self.logger.info(f"Loading {self.keywords_file}")
            self.keywords = np.load(
                open(self.keywords_file, 'rb'), allow_pickle=True)

            self.logger.info(
                "Keywords embeddings file already exists, loading it directly")
            self.logger.info(
                f"Loading {self.keywords_embeddings_file} keywords embeddings")
            self.keywords_embeddings = np.load(
                open(self.keywords_embeddings_file, 'rb'))

    def load_abstract_embeddings(self):
        '''
        https://huggingface.co/sentence-transformers/allenai-specter
        https://huggingface.co/allenai/scibert_scivocab_uncased
        '''

        if self.params.only_create_abstract_embeddings:
            self.logger.info(
                "create abstract embeddings enabled, creating the abstract embeddings from scratch")

            self.logger.info(self.params.name_transformer)
            model = SentenceTransformer(
                self.params.name_transformer).to(self.device)
            abstract_embeddings = []

            self.logger.info(f"batch size: {self.params.batch_size}")

            i = 0
            while i < len(self.abstracts):
                abstracts_batch = [self.abstracts[abstract_id] for abstract_id in list(
                    self.abstracts.keys())[i:min(i+self.params.batch_size, len(self.abstracts))]]
                abstract_level_embeddings = model.encode(
                    abstracts_batch, convert_to_numpy=True, show_progress_bar=False)
                abstract_embeddings.extend(abstract_level_embeddings)
                i += self.params.batch_size

            self.abstract_embeddings = np.array(abstract_embeddings)
            name_file = osp.join(os.getcwd(
            ), "input", f"abstract_embeddings-{self.params.name_transformer.replace('/', '_')}.npy")
            np.save(open(name_file, "wb"), self.abstract_embeddings)

        elif os.path.isfile(self.abstract_embeddings_file):
            self.logger.info(
                "Embedings file already exists, loading it directly")
            self.logger.info(
                f"Loading {self.abstract_embeddings_file} abstract embeddings")
            self.abstract_embeddings = np.load(
                open(self.abstract_embeddings_file, 'rb'))

    def build_predict(self):

        self.logger.info("Building predict ...")
        self.predict_mode = True

        with open(self.path_predict, 'r') as file:
            self.X = []
            for line in file:
                line = line.split(',')
                edg0 = int(line[0])
                edg1 = int(line[1])
                self.X.append((edg0, edg1))

        self.y = np.zeros(len(self.X))

    def build_train(self):

        self.logger.info(f"Building train ...")
        self.predict_mode = False

        nodes = list(self.G.nodes())
        edges = self.G.edges()

        m = self.G.number_of_edges()
        n = self.G.number_of_nodes()

        X = []
        y = []

        self.logger.info("Starting loop ...")

        with get_progress_bar() as progress:
            i = 0
            task1 = progress.add_task(
                f"[cyan]Processing embeddings ", total=m, info="-")
            for i, edge in enumerate(self.G.edges()):

                X.append(edge)
                y.append(1)

                n1 = nodes[randint(0, n-1)]
                n2 = nodes[randint(0, n-1)]

                while (n1, n2) in edges:
                    n1 = nodes[randint(0, n-1)]
                    n2 = nodes[randint(0, n-1)]

                X.append((n1, n2))
                y.append(0)

                progress.update(task1, advance=1, info=f"{i}/{m}")

        self.X = X
        self.y = torch.Tensor(y).float()

        self.logger.info("Finished building train ...")

        self.clean()

    def clean(self):

        self.logger.info("Cleaning useless class attributes ...")

        del self.abstracts

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        raise NotImplementedError(f'Should be implemented in derived class!')
