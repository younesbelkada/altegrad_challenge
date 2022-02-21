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
