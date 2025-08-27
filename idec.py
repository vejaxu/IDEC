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

from scipy.interpolate import Rbf
from sklearn.metrics.pairwise import euclidean_distances


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

    # 创建可视化目录
    vis_dir = f'tsne/{args.dataset}/gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}'
    os.makedirs(vis_dir, exist_ok=True)
    log_path = os.path.join(vis_dir, "log.txt")

    # 保存训练配置
    with open(log_path, "a") as f:
        f.write("========== IDEC Training Configuration ==========\n")
        for arg_name, arg_val in vars(args).items():
            f.write(f"{arg_name}: {arg_val}\n")
        f.write("=================================================\n\n")
        f.write(str(model))
        f.write("=================================================\n\n")

    """ Pretrain Autoencoder """
    model.pretrain()  # 假设内部已处理路径

    # 数据加载
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 初始聚类：使用 K-Means
    data = torch.Tensor(dataset.x).to(device)
    with torch.no_grad():
        x_bar, z = model.ae(data)
    z_np = z.cpu().numpy()

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z_np)
    y = dataset.y
    nmi_k = nmi_score(y_pred, y)
    print(f"Initial NMI: {nmi_k:.4f}")

    # 初始化聚类层
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    y_pred_last = y_pred.copy()

    # 开始训练
    model.train()
    for epoch in range(args.train_epoch):
        if epoch % args.update_interval == 0:
            with torch.no_grad():
                _, q = model(data)
                p = target_distribution(q).data
                y_pred = q.cpu().numpy().argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred.copy()

                acc = cluster_acc(y, y_pred)
                nmi = nmi_score(y, y_pred)
                ari = ari_score(y, y_pred)

            # 记录指标
            log_str = f'Iter {epoch:03d}: Acc {acc:.4f}, NMI {nmi:.4f}, ARI {ari:.4f}, Δlabel {delta_label:.4f}'
            print(log_str)
            with open(log_path, "a") as f:
                f.write(log_str + "\n")

            # ==================== 可视化部分 ====================
            model.eval()
            cluster_centers_z = model.cluster_layer.data  # (k, n_z)

            with torch.no_grad():
                # 获取编码空间表示 z (高维)
                enc_h1 = F.relu(model.ae.enc_1(data))
                enc_h2 = F.relu(model.ae.enc_2(enc_h1))
                enc_h3 = F.relu(model.ae.enc_3(enc_h2))
                z_representation = model.ae.z_layer(enc_h3).cpu().numpy()  # (n_samples, n_z)

                # 获取聚类中心在编码空间的表示 (k, n_z)
                enc_center_h1 = F.relu(model.ae.enc_1(cluster_centers_z))
                enc_center_h2 = F.relu(model.ae.enc_2(enc_center_h1))
                enc_center_h3 = F.relu(model.ae.enc_3(enc_center_h2))
                z_centers = model.ae.z_layer(enc_center_h3).cpu().numpy()  # (k, n_z)

                # 原始空间数据
                x_input = dataset.x_row  # 原始高维数据 (n_samples, d)

            # ===== t-SNE 降维 =====
            # 原始空间降维
            if x_input.shape[1] > 2:
                tsne_input = TSNE(n_components=2, random_state=42)
                X_2d_original = tsne_input.fit_transform(x_input)
            else:
                X_2d_original = x_input

            # 编码空间降维
            if z_representation.shape[1] > 2:
                tsne_z = TSNE(n_components=2, random_state=42)
                X_2d_encoded = tsne_z.fit_transform(z_representation)
                # 降维聚类中心
                z_centers_2d = tsne_z.transform(z_centers)  # 使用相同映射
            else:
                X_2d_encoded = z_representation
                z_centers_2d = z_centers

            # ===== 计算编码空间距离 =====
            from sklearn.metrics.pairwise import euclidean_distances
            dist_to_centers = euclidean_distances(z_representation, z_centers)  # (n, k)
            min_dist_in_z = dist_to_centers.min(axis=1)  # (n,)

            # ===== 保存可视化 =====
            save_idec_visualizations(
                X_2d_original=X_2d_original,
                X_2d_encoded=X_2d_encoded,
                labels=y_pred,
                centers_encoded=z_centers_2d,
                min_dist_in_z=min_dist_in_z,
                epoch=epoch,
                vis_dir=vis_dir
            )

            model.train()

        # 标准 IDEC 损失更新
        if (epoch + 1) % args.update_interval == 0:
            x_bar, q = model(data)
            p = target_distribution(q).detach()
            loss = model.loss_function(p, q, x_bar, data)
        else:
            x_bar, q = model(data)
            loss = model.reconstruction_loss(x_bar, data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def save_idec_visualizations(X_2d_original, X_2d_encoded, labels, centers_encoded, min_dist_in_z, epoch, vis_dir):
    """
    可视化：
    1. 原始空间和编码空间的聚类结果（不带中心）
    2. 编码空间距离 → 原始空间等高线图
    """
    epoch_dir = f"{vis_dir}/epoch_{epoch:03d}"
    os.makedirs(epoch_dir, exist_ok=True)

    # --- 1. 原始空间聚类图 ---
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d_original[:, 0], X_2d_original[:, 1], c=labels, cmap='viridis', alpha=0.7, s=20)
    plt.title(f"Original Space Clustering - Epoch {epoch}")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{epoch_dir}/original_clustering.jpg", dpi=300, bbox_inches='tight')
    plt.close()

    # --- 2. 编码空间聚类图 ---
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d_encoded[:, 0], X_2d_encoded[:, 1], c=labels, cmap='viridis', alpha=0.7, s=20)
    plt.title(f"Encoded Space Clustering - Epoch {epoch}")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{epoch_dir}/encoded_clustering.jpg", dpi=300, bbox_inches='tight')
    plt.close()

    # --- 3. 原始空间 + 编码空间距离等高线 ---
    x_min, x_max = X_2d_original[:, 0].min() - 1, X_2d_original[:, 0].max() + 1
    y_min, y_max = X_2d_original[:, 1].min() - 1, X_2d_original[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # 使用 RBF 插值：将编码空间距离映射到原始空间网格
    try:
        rbf = Rbf(X_2d_original[:, 0], X_2d_original[:, 1], min_dist_in_z, function='multiquadric', smooth=1)
        dist_on_grid = rbf(grid_points[:, 0], grid_points[:, 1])
        dist_on_grid = dist_on_grid.reshape(xx.shape)
    except:
        # 备用：简单最近邻插值
        from scipy.interpolate import griddata
        dist_on_grid = griddata(X_2d_original, min_dist_in_z, (xx, yy), method='linear', fill_value=np.nanmean(min_dist_in_z))
        dist_on_grid = np.nan_to_num(dist_on_grid)

    # 绘图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d_original[:, 0], X_2d_original[:, 1], c=labels, cmap='viridis', alpha=0.7, s=20)
    contour = plt.contour(xx, yy, dist_on_grid, levels=10, colors='black', alpha=0.6, linestyles='--', linewidths=1.2)
    plt.clabel(contour, inline=True, fontsize=8, fmt="%.2f")

    plt.title(f"Original Space with Encoded Distance Contours - Epoch {epoch}")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.colorbar(scatter, label='Cluster Label')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{epoch_dir}/original_with_distance_contours.jpg", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved visualizations for epoch {epoch} at: {epoch_dir}/")
            

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


    elif args.dataset == 'outliers':
        args.n_clusters = 2
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
        args.n_clusters = 4
        args.n_input = 2
        args.pretrain_epoch = 200
        args.train_epoch = 100
        args.n_z = 2
        args.update_interval = 3
        args.pretrain_path = f'data/init_bias/ae_gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}.pkl'
        dataset = CustomDataset(args.dataset)

    print(args)
    train_idec()
