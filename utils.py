# -*- coding: utf-8 -*-
#
# Copyright © dawnranger.
#
# 2018-05-08 10:15 <dawnranger123@gmail.com>
#
# Distributed under terms of the MIT license.
from __future__ import division, print_function
import numpy as np
import torch
from torch.utils.data import Dataset

from scipy.optimize import linear_sum_assignment
from scipy.io import loadmat
import os


def load_mnist(path='./data/mnist.npz'):
    f = np.load(path)

    x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f[
        'x_test'], f['y_test']
    f.close()
    x = np.concatenate((x_train, x_test))
    x_row = x
    y = np.concatenate((y_train, y_test)).astype(np.int32)
    x = x.reshape((x.shape[0], -1)).astype(np.float32)
    x = np.divide(x, 255.)
    print('MNIST samples', x.shape)
    return x, y, x_row


class MnistDataset(Dataset):

    def __init__(self):
        self.x, self.y, self.x_row = load_mnist()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))
    

def load_AC(path='/data/xwj/aaa/clustering/data/AC.mat'):
    f = loadmat(path)
    x_train, y_train = f['data'], f['class']
    x_row = x_train
    x_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0) + 1e-8)
    x = x_train.astype(np.float32)
    y = y_train.flatten().astype(np.int32)
    return x, y, x_row


class ACDataset(Dataset):
    
    def __init__(self):
        self.x, self.y, self.x_row = load_AC()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))
    

def load_4C(path='/data/xwj/aaa/clustering/data/4C.mat'):
    f = loadmat(path)
    x_train, y_train = f['data'], f['class']
    x_row = x_train
    x_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0) + 1e-8)
    x = x_train.astype(np.float32)
    y = y_train.flatten().astype(np.int32)
    return x, y, x_row


class fourCDataset(Dataset):
    
    def __init__(self):
        self.x, self.y, self.x_row = load_4C()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))
    

def load_sparse_3_dense_3_dense_3(path='/data/xwj/aaa/clustering/data/sparse_3_dense_3_dense_3.mat'):
    f = loadmat(path)
    x_train, y_train = f['data'], f['class']
    x_row = x_train
    x_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0) + 1e-8)
    x = x_train.astype(np.float32)
    y = y_train.flatten().astype(np.int32)
    return x, y, x_row


class sparse_3_dense_3_dense_3Dataset(Dataset):
    
    def __init__(self):
        self.x, self.y, self.x_row = load_sparse_3_dense_3_dense_3()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))
    

def load_sparse_8_dense_1_dense_1(path='/data/xwj/aaa/clustering/data/sparse_8_dense_1_dense_1.mat'):
    f = loadmat(path)
    x_train, y_train = f['data'], f['class']
    x_row = x_train
    x_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0) + 1e-8)
    x = x_train.astype(np.float32)
    y = y_train.flatten().astype(np.int32)
    return x, y, x_row


class sparse_8_dense_1_dense_1Dataset(Dataset):
    
    def __init__(self):
        self.x, self.y, self.x_row = load_sparse_8_dense_1_dense_1()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))
    

def load_one_gaussian_10_one_line_5_2(path='/data/xwj/aaa/clustering/data/one_gaussian_10_one_line_5_2.mat'):
    f = loadmat(path)
    x_train, y_train = f['data'], f['class']
    x_row = x_train
    x_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0) + 1e-8)
    x = x_train.astype(np.float32)
    y = y_train.flatten().astype(np.int32)
    return x, y, x_row


class one_gaussian_10_one_line_5_2Dataset(Dataset):
    
    def __init__(self):
        self.x, self.y, self.x_row = load_one_gaussian_10_one_line_5_2()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))
    

def load_sparse_3_dense_3_dense_3_10(path='/data/xwj/aaa/clustering/data/sparse_3_dense_3_dense_3_10.mat'):
    f = loadmat(path)
    x_train, y_train = f['all_data'], f['all_labels']
    x_row = x_train
    x_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0) + 1e-8)
    x = x_train.astype(np.float32)
    y = y_train.flatten().astype(np.int32)
    return x, y, x_row


class sparse_3_dense_3_dense_3_10_Dataset(Dataset):
    
    def __init__(self):
        self.x, self.y, self.x_row = load_sparse_3_dense_3_dense_3_10()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))
    

def load_sparse_8_dense_1_dense_1_10(path='/data/xwj/aaa/clustering/data/sparse_8_dense_1_dense_1_10.mat'):
    f = loadmat(path)
    x_train, y_train = f['all_data'], f['all_labels']
    x_row = x_train
    x_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0) + 1e-8)
    x = x_train.astype(np.float32)
    y = y_train.flatten().astype(np.int32)
    return x, y, x_row


class sparse_8_dense_1_dense_1_10_Dataset(Dataset):
    
    def __init__(self):
        self.x, self.y, self.x_row = load_sparse_8_dense_1_dense_1_10()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))
    

def load_one_gaussian_10_one_line_5_2_10(path='/data/xwj/aaa/clustering/data/one_gaussian_10_one_line_5_2_10.mat'):
    f = loadmat(path)
    x_train, y_train = f['all_data'], f['all_labels']
    x_row = x_train
    x_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0) + 1e-8)
    x = x_train.astype(np.float32)
    y = y_train.flatten().astype(np.int32)
    return x, y, x_row


class one_gaussian_10_one_line_5_2_10Dataset(Dataset):
    
    def __init__(self):
        self.x, self.y, self.x_row = load_one_gaussian_10_one_line_5_2_10()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))


#######################################################
# Evaluate Critiron
#######################################################


# def cluster_acc(y_true, y_pred):
#     """
#     Calculate clustering accuracy. Require scikit-learn installed
# 
#     # Arguments
#         y: true labels, numpy.array with shape `(n_samples,)`
#         y_pred: predicted labels, numpy.array with shape `(n_samples,)`
# 
#     # Return
#         accuracy, in [0,1]
#     """
#     y_true = y_true.astype(np.int64)
#     assert y_pred.size == y_true.size
#     D = max(y_pred.max(), y_true.max()) + 1
#     w = np.zeros((D, D), dtype=np.int64)
#     for i in range(y_pred.size):
#         w[y_pred[i], y_true[i]] += 1
#     from sklearn.utils.linear_assignment_ import linear_assignment
#     ind = linear_assignment(w.max() - w)
#     return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy.

    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    total = sum([w[i, j] for i, j in zip(row_ind, col_ind)])
    return total * 1.0 / y_pred.size
