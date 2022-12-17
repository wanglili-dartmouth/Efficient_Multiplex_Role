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
def test(labels,embedding):

    labels=np.array(labels)
    sss = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
#########################
    i = 1
    f1_micros = []
    f1_macros = []
    for split_train, split_test in sss.split(embedding, labels):

        model=KNeighborsClassifier(n_neighbors=4)
        model.fit(embedding[split_train], labels[split_train])        
        predictions = model.predict(embedding[split_test])
        f1_micro = f1_score(labels[split_test], predictions, average="micro")
        f1_macro = f1_score(labels[split_test], predictions, average="macro")
        f1_micros.append(f1_micro)
        f1_macros.append(f1_macro)
        i += 1
     #   print(f1_macros)
    return np.mean(f1_micros), np.mean(f1_macros)
def main(args):

    args.save_path="wuliao"
    width_basis = 18
    np.random.seed(0)
    basis_type = "cycle_3" 
    nb_shapes = 6 
    list_shapes = [["house1"],["house2"],["house1"],["house2"],["house1"],["house2"]]
    sample = open('house_p.out', 'w') 
    Total_times=10
    ami=[]
    sil=[]
    hom=[]
    comp=[]
    f1_micro=[]
    f1_macro=[]
    random.seed(0)
    np.random.seed(0)
    for times in range(Total_times):
        G, communities, plugins, role_id = build_structure(width_basis, basis_type, list_shapes, start=0,
                            rdm_basis_plugins =False, add_random_edges=7,
                            plot=False, savefig=False)
        graph=G
        G=nx.relabel_nodes(G, lambda x: str(x))
        #print(list(G.nodes()))
        print( 'nb of nodes in the graph: ', G.number_of_nodes())
        print( 'nb of edges in the graph: ', G.number_of_edges())
        nx.write_weighted_edgelist(G, "test3.edgelist")

        graph = network.read_from_edgelist("test3.edgelist", index_from_zero=args.index_from_0)
        network.sort_graph_by_degree(graph)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Graph().as_default(), tf.Session(config=config) as sess, tf.device('/gpu:0'):
            alg = eni(graph, args, sess)
            print("max degree: {}".format(alg.degree_max))
            alg.train()
            alg.save()
            embeddings= alg.get_embeddings()
        
        trans_data=np.array([(embeddings[i]) for i in range(len(role_id))])
        colors = role_id
        nb_clust = len(np.unique(colors))
        km = sk.cluster.AgglomerativeClustering(n_clusters=nb_clust,linkage='complete')
        km.fit(trans_data)
        labels_pred = km.labels_

        labels = colors
        a,b=test(colors,trans_data)
        f1_micro.append(a)
        f1_macro.append(b)
        ami.append(sk.metrics.adjusted_mutual_info_score(colors, labels_pred))
        sil.append(sk.metrics.silhouette_score(trans_data,labels_pred))  
        hom.append(sk.metrics.homogeneity_score(colors, labels_pred))
        comp.append(sk.metrics.completeness_score(colors, labels_pred))
    print ('Homogeneity \t Completeness \t AMI \t nb clusters \t Silhouette \t f1_micro \t f1_macro \n', file = sample)
    print (str(np.mean(hom))+'\t'+str(np.mean(comp))+'\t'+str(np.mean(ami))+'\t'+str(np.mean(nb_clust))+'\t'+str(np.mean(sil))+'\t'+str(np.mean(f1_micro))+'\t'+str(np.mean(f1_macro)), file = sample)
    print (str(np.std(hom))+'\t'+str(np.std(comp))+'\t'+str(np.std(ami))+'\t'+str(np.std(nb_clust))+'\t'+str(np.std(sil))+'\t'+str(np.std(f1_micro))+'\t'+str(np.std(f1_macro)), file = sample)
    
    
    print("Homogeneity:  ",hom, file = sample)
    print("Completeness:  ",comp, file = sample)
    print("AMI:  ",ami, file = sample)
    print("Silhouette:  ",sil, file = sample)
    print("f1_micro:  ",f1_micro, file = sample)
    print("f1_macro:  ",f1_macro, file = sample)
if __name__ == '__main__':
    main(parse_args())
