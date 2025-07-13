# -*- coding: utf-8 -*-
#
# Copyright © dawnranger.
#
# 2018-05-08 10:15 <dawnranger123@gmail.com>
#
# Distributed under terms of the MIT license.
from __future__ import print_function, division
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear

from utils import MnistDataset, cluster_acc

import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import *


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()

        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)

        self.z_layer = Linear(n_enc_3, n_z)

        # decoder
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)

        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):

        # encoder
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))

        z = self.z_layer(enc_h3)

        # decoder
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z


class IDEC(nn.Module):

    def __init__(self,
                 n_enc_1,
                 n_enc_2,
                 n_enc_3,
                 n_dec_1,
                 n_dec_2,
                 n_dec_3,
                 n_input,
                 n_z,
                 n_clusters,
                 alpha=1,
                 pretrain_path='data/ae_mnist.pkl'):
        super(IDEC, self).__init__()
        self.alpha = 1.0
        self.pretrain_path = pretrain_path

        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z)) #  定义并注册一个“可学习的聚类中心矩阵”作为模型参数
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self, path=''):
        if path == '':
            pretrain_ae(self.ae)
        # load pretrain weights
        self.ae.load_state_dict(torch.load(self.pretrain_path))
        print('load pretrained ae from', path)

    def forward(self, x):

        x_bar, z = self.ae(x)
        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x_bar, q


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def pretrain_ae(model):
    '''
    pretrain autoencoder
    '''
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.pretrain_epoch):
        total_loss = 0.
        for batch_idx, (x, _, _) in enumerate(train_loader):
            x = x.to(device)

            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (batch_idx + 1)))
        torch.save(model.state_dict(), args.pretrain_path)
    print("model saved to {}.".format(args.pretrain_path))


def train_idec():
    vis_dir = f'tsne/{args.dataset}/gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}'
    os.makedirs(vis_dir, exist_ok=True)
    log_path = os.path.join(vis_dir, "log.txt")
    with open(log_path, "a") as f:
        f.write("========== IDEC Training Configuration ==========\n")
        for arg_name, arg_val in vars(args).items():
            f.write(f"{arg_name}: {arg_val}\n")
        f.write("=================================================\n\n")


    model = IDEC(
        # n_enc_1=500,
        # n_enc_2=500,
        # n_enc_3=1000,
        # n_dec_1=1000,
        # n_dec_2=500,
        # n_dec_3=500,
        n_enc_1=10,
        n_enc_2=100,
        n_enc_3=200,
        n_dec_1=200,
        n_dec_2=100,
        n_dec_3=10,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        alpha=1.0,
        pretrain_path=args.pretrain_path).to(device)


    """这里在需要调整
    tips
    tips
    tips
    tips
    """
    #  model.pretrain('data/ae_mnist.pkl')
    # model.pretrain(args.pretrain_path)
    model.pretrain()

    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # cluster parameter initiate
    data = dataset.x
    y = dataset.y
    data = torch.Tensor(data).to(device)
    x_bar, hidden = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
    nmi_k = nmi_score(y_pred, y)
    print("initial nmi score={:.4f}".format(nmi_k))

    hidden = None
    x_bar = None

    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    model.train()
    for epoch in range(args.train_epoch):

        if epoch % args.update_interval == 0:

            _, tmp_q = model(data)

            # update target distribution p
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            # evaluate clustering performance
            y_pred = tmp_q.cpu().numpy().argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]
            y_pred_last = y_pred

            acc = cluster_acc(y, y_pred)
            nmi = nmi_score(y, y_pred)
            ari = ari_score(y, y_pred)
            # print('Iter {}'.format(epoch), ':Acc {:.4f}'.format(acc),
            #       ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))
            

            with open(log_path, "a") as f:
                log_str = 'Iter {:03d}: Acc {:.4f}, NMI {:.4f}, ARI {:.4f}, Δlabel {:.4f}\n'.format(
                    epoch, acc, nmi, ari, delta_label)
                print(log_str.strip())
                f.write(log_str)
            

            model.eval()
            with torch.no_grad():
                _, z = model.ae(data)
            z = z.cpu().numpy()
            # 获取原始输入数据
            x_input = dataset.x_row  # numpy array
            # 对原始输入数据做 t-SNE 降维（固定 random_state 保证稳定
            if x_input.shape[1] > 2:
                tsne_input = TSNE(n_components=2, random_state=42)
                x_2d = tsne_input.fit_transform(x_input)
            else:
                x_2d = x_input
            def plot_tsne(X_2d, labels, title, subplot_idx):
                plt.subplot(1, 2, subplot_idx)
                scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.7, s=15)
                plt.title(title)
                plt.xlabel("Dimension 1")
                plt.ylabel("Dimension 2")
                plt.colorbar(scatter, label='Class Labels')
                plt.grid(True)
                plt.axis('equal')
            plt.figure(figsize=(12, 6))
            plot_tsne(x_2d, y, "True Labels", 1)
            plot_tsne(x_2d, y_pred, "IDEC", 2)
            plt.tight_layout()
            plt.savefig(f"{vis_dir}/epoch_{epoch:03d}.jpg", dpi=300)


            if epoch > 0 and delta_label < args.tol:
                print('delta_label {:.4f}'.format(delta_label), '< tol',
                      args.tol)
                print('Reached tolerance threshold. Stopping training.')
                break
        for batch_idx, (x, _, idx) in enumerate(train_loader):

            x = x.to(device)
            idx = idx.to(device)

            x_bar, q = model(x)

            reconstr_loss = F.mse_loss(x_bar, x)
            kl_loss = F.kl_div(q.log(), p[idx])
            loss = args.gamma * kl_loss + reconstr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=7, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--pretrain_path', type=str, default='data/ae_mnist')
    parser.add_argument(
        '--gamma',
        default=0.1,
        type=float,
        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--train_epoch', default=100, type=int)
    parser.add_argument('--pretrain_epoch', default=200, type=int)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda:1" if args.cuda else "cpu")

    if args.dataset == 'mnist':
        args.pretrain_path = f'data/mnist/ae_gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}.pkl'
        args.n_clusters = 10
        args.n_input = 784
        dataset = MnistDataset()


    elif args.dataset == 'AC':
        args.n_clusters = 2
        args.n_input = 2
        args.pretrain_epoch = 200
        args.train_epoch = 100
        args.n_z = 10
        args.update_interval = 3
        args.pretrain_path = f'data/AC/ae_gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}.pkl'
        dataset = ACDataset()


    elif args.dataset == 'sparse_3_dense_3_dense_3':
        args.n_clusters = 3
        args.n_input = 2
        args.pretrain_epoch = 200
        args.train_epoch = 100
        args.n_z = 10
        args.update_interval = 3
        args.pretrain_path = f'data/sparse_3_dense_3_dense_3/ae_gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}.pkl'
        dataset = sparse_3_dense_3_dense_3Dataset()


    elif args.dataset == 'sparse_8_dense_1_dense_1':
        args.n_clusters = 3
        args.n_input = 2
        args.pretrain_epoch = 200
        args.train_epoch = 100
        args.n_z = 10
        args.update_interval = 3
        args.pretrain_path = f'data/sparse_8_dense_1_dense_1/ae_gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}.pkl'
        dataset = sparse_8_dense_1_dense_1Dataset()


    elif args.dataset == 'one_gaussian_10_one_line_5_2':
        args.n_clusters = 2
        args.n_input = 2
        args.pretrain_epoch = 200
        args.train_epoch = 100
        args.n_z = 10
        args.update_interval = 3
        args.pretrain_path = f'data/one_gaussian_10_one_line_5_2/ae_gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}.pkl'
        dataset = one_gaussian_10_one_line_5_2Dataset()


    elif args.dataset == 'sparse_3_dense_3_dense_3_10':
        args.n_clusters = 3
        args.n_input = 2
        args.pretrain_epoch = 200
        args.train_epoch = 100
        args.n_z = 10
        args.update_interval = 10
        args.pretrain_path = f'data/sparse_3_dense_3_dense_3_10/ae_gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}.pkl'
        dataset = sparse_3_dense_3_dense_3_10_Dataset()


    elif args.dataset == 'sparse_8_dense_1_dense_1_10':
        args.n_clusters = 3
        args.n_input = 2
        args.pretrain_epoch = 200
        args.train_epoch = 100
        args.n_z = 10
        args.update_interval = 3
        args.pretrain_path = f'data/sparse_8_dense_1_dense_1_10/ae_gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}.pkl'
        dataset = sparse_8_dense_1_dense_1_10_Dataset()


    elif args.dataset == 'one_gaussian_10_one_line_5_2_10':
        args.n_clusters = 2
        args.n_input = 2
        args.pretrain_epoch = 200
        args.train_epoch = 100
        args.n_z = 10
        args.update_interval = 3
        args.pretrain_path = f'data/one_gaussian_10_one_line_5_2_10/ae_gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}.pkl'
        dataset = one_gaussian_10_one_line_5_2_10Dataset()


    print(args)
    train_idec()
