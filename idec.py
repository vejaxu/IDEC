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
from scipy.spatial.distance import cdist


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
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
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
        os.makedirs(os.path.dirname(args.pretrain_path), exist_ok=True)
        torch.save(model.state_dict(), args.pretrain_path)
    print("model saved to {}.".format(args.pretrain_path))



def train_idec():
    model = IDEC(
        # n_enc_1=100,
        # n_enc_2=100,
        # n_enc_3=200,
        # n_dec_1=200,
        # n_dec_2=100,
        # n_dec_3=100,
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=1000,
        n_dec_1=1000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        alpha=1.0,
        pretrain_path=args.pretrain_path).to(device)
    
    vis_dir = f'tsne/{args.dataset}/gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}'
    os.makedirs(vis_dir, exist_ok=True)
    log_path = os.path.join(vis_dir, "log.txt")

    with open(log_path, "a") as f:
        f.write("========== IDEC Training Configuration ==========\n")
        for arg_name, arg_val in vars(args).items():
            f.write(f"{arg_name}: {arg_val}\n")
        f.write("=================================================\n\n")
        f.write(str(model))
        f.write("=================================================\n\n")

    """ATTENTION !!!"""
    # model.pretrain(args.pretrain_path)
    model.pretrain()

    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # cluster parameter initiate
    data = dataset.x
    y = dataset.y
    data = torch.Tensor(data).to(device)

    """正常流程"""
    x_bar, hidden = model.ae(data)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())

    """init_bias 初始化问题"""
    # x_bar, hidden = model.ae(data)
    # hidden_np = hidden.data.cpu().numpy()
    # # 选择 class_id = 0 的两个点作为初始中心
    # class_id = 1
    # indices = np.where(y == class_id)[0][:2]  # 同一个类的两个点
    # init_centers = hidden_np[indices]
    # kmeans = KMeans(n_clusters=args.n_clusters, init=init_centers, n_init=1)
    # y_pred = kmeans.fit_predict(hidden_np)

    # x_bar, hidden = model.ae(data)
    # hidden_np = hidden.data.cpu().numpy()
    # class_id = 1
    # indices = np.where(y == class_id)[0] 
    # class_embeddings = hidden_np[indices] 
    # dist_matrix = cdist(class_embeddings, class_embeddings, metric='euclidean')
    # max_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    # i, j = max_idx
    # point1_idx_in_class = i
    # point2_idx_in_class = j
    # init_point1_idx = indices[point1_idx_in_class]
    # init_point2_idx = indices[point2_idx_in_class]
    # init_centers = hidden_np[[init_point1_idx, init_point2_idx]]
    # kmeans = KMeans(n_clusters=args.n_clusters, init=init_centers, n_init=1)
    # y_pred = kmeans.fit_predict(hidden_np)

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
            
            with open(log_path, "a") as f:
                log_str = 'Iter {:03d}: Acc {:.4f}, NMI {:.4f}, ARI {:.4f}, Δlabel {:.4f}\n'.format(
                    epoch, acc, nmi, ari, delta_label)
                print(log_str.strip())
                f.write(log_str)
            
            model.eval()
            cluster_centers = model.cluster_layer.data
            
            with torch.no_grad():
                enc_h1 = F.relu(model.ae.enc_1(data))
                enc_h2 = F.relu(model.ae.enc_2(enc_h1))
                enc_h3 = F.relu(model.ae.enc_3(enc_h2))
                x_encoded = model.ae.z_layer(enc_h3).cpu().numpy()

            x_input = dataset.x_row

            if x_input.shape[1] > 2:
                tsne_input = TSNE(n_components=2, random_state=42)
                x_input_2d = tsne_input.fit_transform(x_input)
                x_encoded_2d = tsne_input.fit_transform(x_encoded)
            else:
                x_input_2d = x_input
                x_encoded_2d = x_encoded

            save_simple_visualizations(x_input_2d, x_encoded_2d, y_pred, epoch, vis_dir)
            plot_contour_with_centers(
                x_input=x_input,
                autoencoder=model,
                cluster_centers=cluster_centers,  # 来自 model.cluster_layer.data
                y_pred=y_pred,
                epoch=epoch,
                vis_dir=vis_dir
            )

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


def save_simple_visualizations(X_2d_original, X_2d_encoded, labels, epoch, vis_dir):
    epoch_dir = f"{vis_dir}/epoch_{epoch:03d}"
    os.makedirs(epoch_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d_original[:, 0], X_2d_original[:, 1], c=labels, cmap='viridis', alpha=0.7, s=15)
    plt.title(f"Original Space Clustering - Epoch {epoch}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.colorbar(scatter, label='Cluster Labels')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{epoch_dir}/original_space_clustering_epoch_{epoch:03d}.jpg", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d_encoded[:, 0], X_2d_encoded[:, 1], c=labels, cmap='viridis', alpha=0.7, s=15)
    plt.title(f"Encoded Space Clustering - Epoch {epoch}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.colorbar(scatter, label='Cluster Labels')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{epoch_dir}/encoded_space_clustering_epoch_{epoch:03d}.jpg", dpi=300, bbox_inches='tight')
    plt.close()


def plot_contour_with_centers(
    x_input,
    autoencoder,
    cluster_centers,
    y_pred,
    epoch,
    vis_dir,
    n_grid=100
):
    device = next(autoencoder.parameters()).device
    autoencoder.eval()

    os.makedirs(vis_dir, exist_ok=True)

    if isinstance(x_input, np.ndarray):
        x_input_tensor = torch.tensor(x_input, dtype=torch.float32).to(device)
    else:
        x_input_tensor = x_input.to(device)

    if isinstance(cluster_centers, np.ndarray):
        cluster_centers = torch.tensor(cluster_centers, dtype=torch.float32).to(device)
    else:
        cluster_centers = cluster_centers.to(device)

    assert x_input.shape[1] >= 2, "Input dimension must be at least 2 for 2D contour plot."

    x0_min, x0_max = x_input[:, 0].min(), x_input[:, 0].max()
    x1_min, x1_max = x_input[:, 1].min(), x_input[:, 1].max()

    x0_range = np.linspace(x0_min - 0.5, x0_max + 0.5, n_grid)
    x1_range = np.linspace(x1_min - 0.5, x1_max + 0.5, n_grid)
    xx, yy = np.meshgrid(x0_range, x1_range)
    grid_points_2d = np.c_[xx.ravel(), yy.ravel()]

    if x_input.shape[1] > 2:
        rest_mean = np.mean(x_input, axis=0, keepdims=True)[:, 2:]
        rest_repeated = np.tile(rest_mean, (grid_points_2d.shape[0], 1))
        grid_full = np.hstack([grid_points_2d, rest_repeated])
    else:
        grid_full = grid_points_2d

    grid_tensor = torch.tensor(grid_full, dtype=torch.float32).to(device)

    with torch.no_grad():
        _, z_grid = autoencoder.ae(grid_tensor)
        z_grid_np = z_grid.cpu().numpy()

        _, x_encoded = autoencoder.ae(x_input_tensor)
        x_encoded_np = x_encoded.cpu().numpy()

        centers_np = cluster_centers.cpu().numpy()

    print(f"🔍 Computing distances for {z_grid_np.shape[0]} grid points to {centers_np.shape[0]} cluster centers...")

    distances = cdist(z_grid_np, centers_np, metric='euclidean')

    min_distances = np.min(distances, axis=1)
    zz = min_distances.reshape(xx.shape)

    plt.figure(figsize=(8, 6))

    contour = plt.contour(xx, yy, zz, levels=10, colors='black', alpha=0.6, linewidths=0.8)
    plt.clabel(contour, inline=True, fontsize=8, fmt="%.2f")

    contourf = plt.contourf(xx, yy, zz, levels=50, cmap='RdYlBu_r', alpha=0.7)
    plt.colorbar(contourf, label='Min Distance to Cluster Centers (in latent space)')

    scatter = plt.scatter(
        x_input[:, 0], x_input[:, 1],
        c=y_pred, cmap='Set3', s=20, edgecolors='k', alpha=0.8
    )
    plt.title(f'Contour Map of Latent Distance - Epoch {epoch}')
    plt.xlabel('Input Dim 1')
    plt.ylabel('Input Dim 2')
    plt.tight_layout()

    save_path = f"{vis_dir}/contour_epoch_{epoch:03d}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"✅ Contour plot saved to {save_path}")


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
    device = torch.device("cuda:0" if args.cuda else "cpu")

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
        args.n_z = 2
        args.update_interval = 3
        args.pretrain_path = f'data/AC/ae_gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}.pkl'
        dataset = CustomDataset(args.dataset)


    elif args.dataset == '4C':
        args.n_clusters = 4
        args.n_input = 2
        args.pretrain_epoch = 200
        args.train_epoch = 100
        args.n_z = 2
        args.update_interval = 3
        args.pretrain_path = f'data/4C/ae_gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}.pkl'
        dataset = CustomDataset(args.dataset)


    elif args.dataset == 'sparse_3_dense_3_dense_3':
        args.n_clusters = 3
        args.n_input = 2
        args.pretrain_epoch = 200
        args.train_epoch = 100
        args.n_z = 2
        args.update_interval = 3
        args.pretrain_path = f'data/sparse_3_dense_3_dense_3/ae_gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}.pkl'
        dataset = CustomDataset(args.dataset)


    elif args.dataset == 'sparse_8_dense_1_dense_1':
        args.n_clusters = 3
        args.n_input = 2
        args.pretrain_epoch = 200
        args.train_epoch = 100
        args.n_z = 2
        args.update_interval = 3
        args.pretrain_path = f'data/sparse_8_dense_1_dense_1/ae_gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}.pkl'
        dataset = CustomDataset(args.dataset)


    elif args.dataset == 'one_gaussian_10_one_line_5_2':
        args.n_clusters = 2
        args.n_input = 2
        args.pretrain_epoch = 200
        args.train_epoch = 100
        args.n_z = 2
        args.update_interval = 3
        args.pretrain_path = f'data/one_gaussian_10_one_line_5_2/ae_gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}.pkl'
        dataset = CustomDataset(args.dataset)


    elif args.dataset == 'sparse_3_dense_3_dense_3_10':
        args.n_clusters = 3
        args.n_input = 2
        args.pretrain_epoch = 200
        args.train_epoch = 100
        args.n_z = 2
        args.update_interval = 10
        args.pretrain_path = f'data/sparse_3_dense_3_dense_3_10/ae_gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}.pkl'
        dataset = CustomDataset(args.dataset)


    elif args.dataset == 'sparse_8_dense_1_dense_1_10':
        args.n_clusters = 3
        args.n_input = 2
        args.pretrain_epoch = 200
        args.train_epoch = 100
        args.n_z = 2
        args.update_interval = 3
        args.pretrain_path = f'data/sparse_8_dense_1_dense_1_10/ae_gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}.pkl'
        dataset = CustomDataset(args.dataset)


    elif args.dataset == 'one_gaussian_10_one_line_5_2_10':
        args.n_clusters = 2
        args.n_input = 2
        args.pretrain_epoch = 200
        args.train_epoch = 100
        args.n_z = 2
        args.update_interval = 3
        args.pretrain_path = f'data/one_gaussian_10_one_line_5_2_10/ae_gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}.pkl'
        dataset = CustomDataset(args.dataset)


    elif args.dataset == 'overlapping':
        args.n_clusters = 2
        args.n_input = 2
        args.pretrain_epoch = 200
        args.train_epoch = 100
        args.n_z = 2
        args.update_interval = 3
        args.pretrain_path = f'data/overlapping/ae_gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}.pkl'
        dataset = CustomDataset(args.dataset)


    elif args.dataset == 'non_spherical':
        args.n_clusters = 2
        args.n_input = 2
        args.pretrain_epoch = 200
        args.train_epoch = 100
        args.n_z = 2
        args.update_interval = 3
        args.pretrain_path = f'data/non_spherical/ae_gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}.pkl'
        dataset = CustomDataset(args.dataset)


    elif args.dataset == 'non_spherical_2':
        args.n_clusters = 2
        args.n_input = 2
        args.pretrain_epoch = 200
        args.train_epoch = 100
        args.n_z = 2
        args.update_interval = 3
        args.pretrain_path = f'data/non_spherical_2/ae_gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}.pkl'
        dataset = CustomDataset(args.dataset)

    
    elif args.dataset == 'non_spherical_3':
        args.n_clusters = 2
        args.n_input = 2
        args.pretrain_epoch = 200
        args.train_epoch = 100
        args.n_z = 2
        args.update_interval = 3
        args.pretrain_path = f'data/non_spherical_3/ae_gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}.pkl'
        dataset = CustomDataset(args.dataset)


    elif args.dataset == 'outliers':
        args.n_clusters = 3
        args.n_input = 2
        args.pretrain_epoch = 200
        args.train_epoch = 100
        args.n_z = 2
        args.update_interval = 3
        args.pretrain_path = f'data/outliers/ae_gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}.pkl'
        dataset = CustomDataset(args.dataset)


    elif args.dataset == 'metric_mismatch':
        args.n_clusters = 2
        args.n_input = 2
        args.pretrain_epoch = 200
        args.train_epoch = 100
        args.n_z = 2
        args.update_interval = 3
        args.pretrain_path = f'data/metric_mismatch/ae_gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}.pkl'
        dataset = CustomDataset(args.dataset)


    elif args.dataset == 'init_bias':
        args.n_clusters = 2
        args.n_input = 2
        args.pretrain_epoch = 200
        args.train_epoch = 100
        args.n_z = 2
        args.update_interval = 3
        args.pretrain_path = f'data/init_bias/ae_gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}.pkl'
        dataset = CustomDataset(args.dataset)

    elif args.dataset == 'imbalanced':
        args.n_clusters = 3
        args.n_input = 2
        args.pretrain_epoch = 200
        args.train_epoch = 100
        args.n_z = 2
        args.update_interval = 3
        args.pretrain_path = f'data/imbalanced/ae_gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}.pkl'
        dataset = CustomDataset(args.dataset)

    print(args)
    train_idec()
