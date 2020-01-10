import os
import sys
import numpy as np
import pandas as pd
from scipy import sparse


# DATA_DIR = 'thisismyjam-datadump'

def load_dimension(DATA_DIR):
    sid_file = os.path.join(DATA_DIR, 'item2id.txt')
    n_items = sum(1 for line in open(sid_file))

    uid_file = os.path.join(DATA_DIR, 'profile2id.txt')
    n_users = sum(1 for line in open(uid_file))

    return n_items, n_users

def drop_by_user(arr, drop_percent, shape):
    arr = arr.toarray()
    for row in range(0, len(arr)):
        s = np.sum(arr[row])
        drop_num = int(drop_percent*s)
        indices = np.random.choice(int(s), drop_num, replace=False)
        count = 0
        for col in range(len(arr[0])):
            if arr[row,col] == 1:
                if count in indices:
                    arr[row,col] = 0
                count += 1
    return sparse.csr_matrix(arr, dtype='float64', shape=shape)


def load_train_data(DATA_DIR, n_items, n_users, drop_percent):
    tp = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    n_users = tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    # print(max(rows),max(cols),n_users,n_items)
    data = sparse.csr_matrix((np.ones_like(rows),
                              (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))

    return drop_by_user(data, drop_percent, (n_users, n_items))


def load_user_data(DATA_DIR, n_users):
    data = np.load(os.path.join(DATA_DIR, 'user_p.npy'))
    data = np.clip(data, 0, 1)
    return data


def load_vad_tr_te_data(DATA_DIR, n_items, n_users):
    tp_tr = pd.read_csv(os.path.join(DATA_DIR, 'validation_tr.csv'))
    tp_te = pd.read_csv(os.path.join(DATA_DIR, 'validation_te.csv'))
    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                 (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                 (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te


def load_te_tr_te_data(DATA_DIR, n_items, n_users):
    tp_tr = pd.read_csv(os.path.join(DATA_DIR, 'test_tr.csv'))
    tp_te = pd.read_csv(os.path.join(DATA_DIR, 'test_te.csv'))
    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                 (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                 (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te


def load_rating_matrix(DATA_DIR, n_items, n_users):
    tp = pd.read_csv(os.path.join(DATA_DIR, 'all_ratings.csv'))
    n_users = tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    # print(max(rows),max(cols),n_users,n_items)
    data = sparse.csr_matrix((np.ones_like(rows),
                              (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    return data.toarray().astype('float32')

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sparse.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sparse.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)
    #return sparse_to_tuple(adj_normalized)

def load_adj_matrix(DATA_DIR, drop_percent):
    tp = pd.read_csv(os.path.join(DATA_DIR, 'user_link.csv'))
    n_users = tp['user1'].max() + 1

    rows, cols = tp['user1'], tp['user2']
    # print(max(rows),max(cols),n_users,n_items)
    data = sparse.coo_matrix((np.ones_like(rows),
                             (rows, cols)), dtype='float32',
                             shape=(n_users, n_users))
    data = drop_by_user(data, drop_percent, (n_users, n_users))
    return preprocess_adj(data)