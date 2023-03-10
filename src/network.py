from collections import defaultdict
from functools import reduce
import numpy as np
import networkx as nx
import random
import os, sys
import tensorflow as tf
import six
import utils

def read_from_edgelist(filename, undirected=True, index_from_zero=True):
    line_num = 0
    graph = []
    with open(filename, 'r') as f:
        
        for line in f:
            ls = line.strip().split()
            a, b, w = int(ls[0])+int(index_from_zero), int(ls[1])+int(index_from_zero), int(ls[2])
            while len(graph)-1 < max(a, b):
                graph.append([[],[],[],[]])
            if undirected:
                graph[a][w].append(b)
            graph[b][w].append(a)
            line_num += 1
            
    for i in range(len(graph)):
        for w in range(4):
            graph[i][w]=list(set(graph[i][w]))
    return graph

def get_degree(graph):
    degree = [ [[],[],[],[]] for _ in range(len(graph))]
    for i in range(len(graph)):
        for w in range(4):
            degree[i][w] = len(graph[i][w])
    return degree

def sort_graph_by_degree(graph):
    degree = get_degree(graph)
    for i in range(len(graph)):
        for w in range(4):
            graph[i][w] = sorted(graph[i][w], key=lambda x: degree[x][w])
    return graph

def get_max_degree(graph):
    maxn=0
    for i in range(len(graph)):
        for w in range(4):
            if(len(graph[i][w])>maxn):
                maxn=len(graph[i][w])
    return maxn

def neighborhood_sum(*a):
    return list(map(lambda x: reduce(lambda y, z: y+z, x), zip(*a)))

def p_by_degree(x, graph):
    res=[[],[],[],[]]
    for w in range(4):
        res[w] = np.array([len(graph[i][w])+1e-2 for i in graph[x][w]], dtype=float)
    for w in range(4):
        res[w]/=np.sum(res[w])
    return res


def sampling_with_order(x, sampling=False, sampling_size=50, p=None, padding=False):
    if sampling and len(x) > sampling_size:
        indices = np.random.choice(len(x), sampling_size, replace=False, p=p)
        return [x[i] for i in sorted(indices)], sampling_size
    if padding and len(x) < sampling_size:
        return x+[0]*(sampling_size-len(x)), len(x)
    return x, sampling_size

def get_neighborhood_list(node, graph, sampling=True, sampling_size=100):
    p = p_by_degree(node, graph)
    nodes1, seqlen1 = sampling_with_order(graph[node][1], sampling, sampling_size, p=p[1], padding=True)
    nodes2, seqlen2 = sampling_with_order(graph[node][2], sampling, sampling_size, p=p[2], padding=True)
    nodes3, seqlen3 = sampling_with_order(graph[node][3], sampling, sampling_size, p=p[3], padding=True)
    return node, nodes1, seqlen1, nodes2, seqlen2 ,nodes3, seqlen3 


def degree_to_class(degree, degree_max, nums_class):
    ind = int(np.log2(degree)/np.log2(degree_max+1)*nums_class)
    res = np.zeros(nums_class)
    res[ind] = 1
    return res

def next_batch(graph, degree_max=None, sampling=False, sampling_size=100):
    if degree_max is None:
        degree_max = get_max_degree(graph)
    while True:
        l = np.random.permutation(range(1, len(graph)))
        for i in l:
            #yield (get_neighborhood(i, K, graph, padding=False, sampling=sampling, sampling_size=sampling_size), np.log(len(graph[i])))
            if len(graph[i]) == 0:
                continue
            yield (get_neighborhood_list(i, graph, sampling=sampling, sampling_size=sampling_size), np.log(len(graph[i][1])+1), np.log(len(graph[i][2])+1), np.log(len(graph[i][3])+1)  )

def read_network(filename, undirected=True):
    if not undirected:
        return nx.read_edgelist(filename, nodetype=int, create_using=nx.DiGraph())
    return nx.read_edgelist(filename, nodetype=int)

def get_centrality(network, type_='degree', undirected=True):
    if type_ == 'degree':
        if undirected:
            return nx.degree_centrality(network)
        else:
            return nx.in_degree_centrality(network)
    elif type_ == 'closeness':
        return nx.closeness_centrality(network)
    elif type_ == 'betweenness':
        return nx.betweenness_centrality(network)
    elif type_ == 'eigenvector':
        return nx.eigenvector_centrality(network)
    elif type_ == 'kcore':
        return nx.core_number(network)
    else:
        return None

def save_centrality(data, file_path, type_):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    print("\n".join(map(str, data.values())), file=open(os.path.join(file_path, "{}.index".format(type_)), 'w'))

def load_centrality(file_path, type_):
    filename = os.path.join(file_path, "{}.index".format(type_))
    data = np.loadtxt(filename)
    if type_ == 'spread_number':
        return data
    return np.vstack((np.arange(len(data)), data)).T

def save_to_tsv(labels, file_path, degree_max):
    res = []
    for type_ in labels:
        data = load_centrality(file_path, type_)
        if type_ == 'degree':
            data *= degree_max
        res.append(data)
    res = np.vstack(res).astype(int).T.tolist()
    with open(os.path.join(file_path, 'index.tsv'), 'w') as f:
        print("\t".join(labels), file=f)
        for data in res:
            print("\t".join(list(map(str, data))), file=f)
    return res

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    undirected = sys.argv[2] == "True"
    print("undirected", undirected)
    G = read_network('dataset/{}.edgelist'.format(dataset_name), undirected)
    G.remove_edges_from(G.selfloop_edges())
    graph = read_from_edgelist('dataset/{}.edgelist'.format(dataset_name), index_from_zero=True, undirected=undirected)
    graph = sort_graph_by_degree(graph)
    data = get_neighborhood_list(2, graph)
    print(data)
    save_path = 'result/{}/data'.format(dataset_name)
    centrality = ['degree', 'closeness', 'betweenness', 'eigenvector', 'kcore']
    centrality = ['degree', 'kcore']
    for c in centrality:
        save_centrality(get_centrality(G, c), save_path, c)
    #save_to_tsv(['degree', 'kcore'], save_path, get_max_degree(graph))
    print(load_centrality(save_path, 'degree'))
