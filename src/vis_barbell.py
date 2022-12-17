import argparse
import numpy as np
import time
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
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
import matplotlib
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def plot(embeddings):
    X=[]
    Y=[]
    Label=[]
    for i in range(0,25):
        X.append(embeddings[i][0])
        Y.append(embeddings[i][1])
    canvas_height = 15
    canvas_width = 15
    dot_size = 4500
    text_size = 18
    legend_setting = False #“brief” / “full” / False


    sns.set(style="whitegrid")

    # set canvas height & width
    plt.figure(figsize=(canvas_width, canvas_height))


    color_paltette=[(153,51,255),(0,176,240),(102,255,204),(204,255,153),(255,0,0),(255,255,0),(255,192,0)]
    pts_colors=list(range(25))
    for i in range(25):
        if(i==0):
            pts_colors[i]="color_0"
        if(i==1 or i==2 or i==3 or i==4 or i==6 or i==7 or i==8 or i==9):
            pts_colors[i]="color_1"
        if(i==5 or i==10):
            pts_colors[i]="color_2"
        if(i==11 or i==12):
            pts_colors[i]="color_3"
            
            
        if(i==13 or i==14 or i==15 or i==16 or i==18 or i==19 or i==20 or i==21):
            pts_colors[i]="color_4"
        if(i==17 or i==22):
            pts_colors[i]="color_5"
        if(i==23 or i==24):
            pts_colors[i]="color_6"


    for i in range(7):
        color_paltette[i] = (color_paltette[i][0] / 255, color_paltette[i][1] / 255, color_paltette[i][2] / 255)
        
        
    # reorganize dataset
    draw_dataset = {'x': X,
                    'y': Y, 
                    'label':list(range(1, 25 + 1)),
                    'ptsize': dot_size,
                    "cpaltette": color_paltette,
                    'colors':pts_colors}

    #draw scatterplot points
    ax = sns.scatterplot(x = "x",y = "y", alpha = 1,s = draw_dataset["ptsize"],hue="colors", palette=draw_dataset["cpaltette"], legend = legend_setting, data = draw_dataset)


    return ax
def parse_args():
    parser = argparse.ArgumentParser("Deep Recursive Network Embedding with Regular Equivalence")
    parser.add_argument('--data_path', type=str, help='Directory to load data.')
    parser.add_argument('--save_path', type=str, help='Directory to save data.')
    parser.add_argument('--save_suffix', type=str, default='eni', help='Directory to save data.')
    parser.add_argument('-s', '--embedding_size', type=int, default=2, help='the embedding dimension size')
    parser.add_argument('-e', '--epochs_to_train', type=int, default=200, help='Number of epoch to train. Each epoch processes the training data once completely')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Number of training examples processed per step')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0025, help='initial learning rate')
    parser.add_argument('--undirected', type=bool, default=True, help='whether it is an undirected graph')
    parser.add_argument('-a', '--alpha', type=float, default=0.0, help='the rate of structure loss and orth loss')
    parser.add_argument('-l', '--lamb', type=float, default=0.5, help='the rate of structure loss and guilded loss')
    parser.add_argument('-g', '--grad_clip', type=float, default=5.0, help='clip gradients')
    parser.add_argument('-K', type=int, default=1, help='K-neighborhood')
    parser.add_argument('--sampling_size', type=int, default=100, help='sample number')
    parser.add_argument('--seed', type=int, default=-1, help='random seed')
    return parser.parse_args()

def main(args):
    np.random.seed(int(time.time()) if args.seed == -1 else args.seed)
    args.save_path="wuliao"
    graph = network.read_from_edgelist("dataset/barbell.edgelist",index_from_zero=True)
    network.sort_graph_by_degree(graph)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config) as sess, tf.device('/gpu:0'):
        alg = eni(graph, args, sess)
        print("max degree: {}".format(alg.degree_max))
        alg.train()
        alg.save()
        embeddings= alg.get_embeddings()
    matplotlib.use('Agg')
    ax=plot(embeddings)
    ax.axis("equal")
    ax.figure.savefig("barbell_HSE.pdf",bbox_inches='tight')
if __name__ == '__main__':
    main(parse_args())
