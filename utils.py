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
    

def load_dataset(dataset_name):
    if dataset_name == "AC":
        path='/home/xwj/aaa/clustering/data/AC.mat'
        f = loadmat(path)
        x_train, y_train = f['data'], f['class']
        x_row = x_train
        x_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0) + 1e-8)
        x = x_train.astype(np.float32)
        y = y_train.flatten().astype(np.int32)
        return x, y, x_row
    elif dataset_name == "4C":
        path='/home/xwj/aaa/clustering/data/4C.mat'
        f = loadmat(path)
        x_train, y_train = f['data'], f['class']
        x_row = x_train
        x_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0) + 1e-8)
        x = x_train.astype(np.float32)
        y = y_train.flatten().astype(np.int32)
        return x, y, x_row
    elif dataset_name == "sparse_8_dense_1_dense_1":
        path='/home/xwj/aaa/clustering/data/sparse_8_dense_1_dense_1.mat'
        f = loadmat(path)
        x_train, y_train = f['data'], f['class']
        x_row = x_train
        x_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0) + 1e-8)
        x = x_train.astype(np.float32)
        y = y_train.flatten().astype(np.int32)
        return x, y, x_row
    elif dataset_name == "sparse_3_dense_3_dense_3":
        path='/home/xwj/aaa/clustering/data/sparse_3_dense_3_dense_3.mat'
        f = loadmat(path)
        x_train, y_train = f['data'], f['class']
        x_row = x_train
        x_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0) + 1e-8)
        x = x_train.astype(np.float32)
        y = y_train.flatten().astype(np.int32)
        return x, y, x_row
    elif dataset_name == "overlapping":
        path='/home/xwj/aaa/clustering/data/kmeans/dataset_overlapping.mat'
        f = loadmat(path)
        x_train, y_train = f['data'], f['class']
        x_row = x_train
        x_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0) + 1e-8)
        x = x_train.astype(np.float32)
        y = y_train.flatten().astype(np.int32)
        return x, y, x_row
    elif dataset_name == "non_spherical":
        path='/home/xwj/aaa/clustering/data/kmeans/dataset_non_spherical.mat'
        f = loadmat(path)
        x_train, y_train = f['data'], f['class']
        x_row = x_train
        x_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0) + 1e-8)
        x = x_train.astype(np.float32)
        y = y_train.flatten().astype(np.int32)
        return x, y, x_row
    elif dataset_name == "non_spherical_2":
        path='/home/xwj/aaa/clustering/data/kmeans/dataset_non_spherical_2.mat'
        f = loadmat(path)
        x_train, y_train = f['data'], f['class']
        x_row = x_train
        x_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0) + 1e-8)
        x = x_train.astype(np.float32)
        y = y_train.flatten().astype(np.int32)
        return x, y, x_row
    elif dataset_name == "non_spherical_3":
        path='/home/xwj/aaa/clustering/data/kmeans/dataset_non_spherical_3.mat'
        f = loadmat(path)
        x_train, y_train = f['data'], f['class']
        x_row = x_train
        x_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0) + 1e-8)
        x = x_train.astype(np.float32)
        y = y_train.flatten().astype(np.int32)
        return x, y, x_row
    elif dataset_name == "outliers":
        path='/home/xwj/aaa/clustering/data/kmeans/dataset_outliers.mat'
        f = loadmat(path)
        x_train, y_train = f['data'], f['class']
        x_row = x_train
        x_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0) + 1e-8)
        x = x_train.astype(np.float32)
        y = y_train.flatten().astype(np.int32)
        return x, y, x_row
    elif dataset_name == "metric_mismatch":
        path='/home/xwj/aaa/clustering/data/kmeans/dataset_metric_mismatch.mat'
        f = loadmat(path)
        x_train, y_train = f['data'], f['class']
        x_row = x_train
        x_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0) + 1e-8)
        x = x_train.astype(np.float32)
        y = y_train.flatten().astype(np.int32)
        return x, y, x_row
    elif dataset_name == "init_bias":
        path='/home/xwj/aaa/clustering/data/kmeans/dataset_init_bias.mat'
        f = loadmat(path)
        x_train, y_train = f['data'], f['class']
        x_row = x_train
        x_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0) + 1e-8)
        x = x_train.astype(np.float32)
        y = y_train.flatten().astype(np.int32)
        return x, y, x_row
    elif dataset_name == "imbalanced":
        path='/home/xwj/aaa/clustering/data/kmeans/dataset_imbalanced.mat'
        f = loadmat(path)
        x_train, y_train = f['data'], f['class']
        x_row = x_train
        x_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0) + 1e-8)
        x = x_train.astype(np.float32)
        y = y_train.flatten().astype(np.int32)
        return x, y, x_row
    
class CustomDataset(Dataset):
    def __init__(self, dataset_name):
            self.x, self.y, self.x_row = load_dataset(dataset_name)

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
