import argparse
import numpy as np
import time

import tensorflow as tf

import network
from eni import eni
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import networkx as nx 
import numpy as np
import pandas as pd
import pickle
import random
import seaborn as sb
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sys
from sklearn.neighbors import KNeighborsClassifier
from shapes.shapes import *
from sklearn.metrics import pairwise_distances
from shapes.build_graph import *
from math import sqrt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json

def parse_args():
    parser = argparse.ArgumentParser("Deep Recursive Network Embedding with Regular Equivalence")
    parser.add_argument('--data_path', type=str, help='Directory to load data.')
    parser.add_argument('--save_path', type=str, help='Directory to save data.')
    parser.add_argument('--save_suffix', type=str, default='eni', help='Directory to save data.')
    parser.add_argument('-s', '--embedding_size', type=int, default=128, help='the embedding dimension size')
    parser.add_argument('-e', '--epochs_to_train', type=int, default=1000, help='Number of epoch to train. Each epoch processes the training data once completely')
    parser.add_argument('-b', '--batch_size', type=int, default=256, help='Number of training examples processed per step')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0025, help='initial learning rate')
    parser.add_argument('--undirected', type=bool, default=True, help='whether it is an undirected graph')
    parser.add_argument('-a', '--alpha', type=float, default=0.0, help='the rate of structure loss and orth loss')
    parser.add_argument('-l', '--lamb', type=float, default=0.5, help='the rate of structure loss and guilded loss')
    parser.add_argument('-g', '--grad_clip', type=float, default=5.0, help='clip gradients')
    parser.add_argument('-K', type=int, default=1, help='K-neighborhood')
    parser.add_argument('--sampling_size', type=int, default=100, help='sample number')
    parser.add_argument('--seed', type=int, default=-1, help='random seed')
    parser.add_argument('--index_from_0', type=bool, default=True, help='whether the node index is from zero')
    return parser.parse_args()

def main(args):

    args.save_path="wuliao"
    
    random.seed(0)
    np.random.seed(0)


    graph = network.read_from_edgelist("dataset/graph.edgelist", undirected=False,index_from_zero=args.index_from_0)
    network.sort_graph_by_degree(graph)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    print("start training-----")
    with tf.Graph().as_default(), tf.Session(config=config) as sess, tf.device('/gpu:0'):
        alg = eni(graph, args, sess)
        print("max degree: {}".format(alg.degree_max))
        alg.train()
        alg.save()
        embeddings= alg.get_embeddings()
    
    
if __name__ == '__main__':
    main(parse_args())
