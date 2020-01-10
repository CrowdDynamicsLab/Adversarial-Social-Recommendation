import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import pandas as pd
import os


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    # load the data: x, tx, allx, graph
    # names = ['x', 'tx', 'allx', 'graph']
    # objects = []
    # for i in range(len(names)):
    #     with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
    #         if sys.version_info > (3, 0):
    #             objects.append(pkl.load(f, encoding='latin1'))
    #         else:
    #             objects.append(pkl.load(f))
    # x, tx, allx, graph = tuple(objects)
    # test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    # test_idx_range = np.sort(test_idx_reorder)

    # if dataset == 'citeseer':
    #     # Fix citeseer dataset (there are some isolated nodes in the graph)
    #     # Find isolated nodes, add them as zero-vecs into the right position
    #     test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
    #     tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    #     tx_extended[test_idx_range-min(test_idx_range), :] = tx
    #     tx = tx_extended

    # features = sp.vstack((allx, tx)).tolil()
    # features[test_idx_reorder, :] = features[test_idx_range, :]
    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    DATA_DIR = "/Users/Adit/NCF-GAN-social/epinions_dataset/"
    tp = pd.read_csv(os.path.join(DATA_DIR, 'user_link.csv'))
    n_users = tp['user1'].max() + 1

    rows, cols = tp['user1'], tp['user2']
    # print(max(rows),max(cols),n_users,n_items)
    data = sp.coo_matrix((np.ones_like(rows),
                             (rows, cols)), dtype='float32',
                             shape=(n_users, n_users))
    adj = data
    features = data
    
    # print adj,features
    # exit(0)

    return adj, features

