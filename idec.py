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

from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap


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
        os.makedirs(os.path.dirname(args.pretrain_path), exist_ok=True)
        torch.save(model.state_dict(), args.pretrain_path)
    print("model saved to {}.".format(args.pretrain_path))


# def train_idec():
#     model = IDEC(
#         n_enc_1=500,
#         n_enc_2=500,
#         n_enc_3=1000,
#         n_dec_1=1000,
#         n_dec_2=500,
#         n_dec_3=500,
#         # n_enc_1=10,
#         # n_enc_2=100,
#         # n_enc_3=200,
#         # n_dec_1=200,
#         # n_dec_2=100,
#         # n_dec_3=10,
#         n_input=args.n_input,
#         n_z=args.n_z,
#         n_clusters=args.n_clusters,
#         alpha=1.0,
#         pretrain_path=args.pretrain_path).to(device)
    

#     vis_dir = f'tsne/{args.dataset}/gamma_{args.gamma}_nz_{args.n_z}_update_{args.update_interval}'
#     os.makedirs(vis_dir, exist_ok=True)
#     log_path = os.path.join(vis_dir, "log.txt")
#     with open(log_path, "a") as f:
#         f.write("========== IDEC Training Configuration ==========\n")
#         for arg_name, arg_val in vars(args).items():
#             f.write(f"{arg_name}: {arg_val}\n")
#         f.write("=================================================\n\n")
#         f.write(str(model))
#         f.write("=================================================\n\n")

#     """ATTENTION !!!"""
#     model.pretrain(args.pretrain_path)
#     # model.pretrain()

#     train_loader = DataLoader(
#         dataset, batch_size=args.batch_size, shuffle=False)
#     optimizer = Adam(model.parameters(), lr=args.lr)

#     # cluster parameter initiate
#     data = dataset.x
#     y = dataset.y
#     data = torch.Tensor(data).to(device)
#     x_bar, hidden = model.ae(data)

#     kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
#     y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
#     nmi_k = nmi_score(y_pred, y)
#     print("initial nmi score={:.4f}".format(nmi_k))

#     hidden = None
#     x_bar = None

#     y_pred_last = y_pred
#     model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

#     model.train()
#     for epoch in range(args.train_epoch):

#         if epoch % args.update_interval == 0:

#             _, tmp_q = model(data)

#             # update target distribution p
#             tmp_q = tmp_q.data
#             p = target_distribution(tmp_q)

#             # evaluate clustering performance
#             y_pred = tmp_q.cpu().numpy().argmax(1)
#             delta_label = np.sum(y_pred != y_pred_last).astype(
#                 np.float32) / y_pred.shape[0]
#             y_pred_last = y_pred

#             acc = cluster_acc(y, y_pred)
#             nmi = nmi_score(y, y_pred)
#             ari = ari_score(y, y_pred)
            

#             with open(log_path, "a") as f:
#                 log_str = 'Iter {:03d}: Acc {:.4f}, NMI {:.4f}, ARI {:.4f}, Δlabel {:.4f}\n'.format(
#                     epoch, acc, nmi, ari, delta_label)
#                 print(log_str.strip())
#                 f.write(log_str)
            
#             model.eval()
#             cluster_centers = model.cluster_layer.data
#             with torch.no_grad():
#                 dec_h1 = F.relu(model.ae.dec_1(cluster_centers))
#                 dec_h2 = F.relu(model.ae.dec_2(dec_h1))
#                 dec_h3 = F.relu(model.ae.dec_3(dec_h2))
#                 x_centers = model.ae.x_bar_layer(dec_h3).cpu().numpy()
#             x_input = dataset.x_row
#             if x_input.shape[1] > 2:
#                 tsne_input = TSNE(n_components=2, random_state=42)
#                 x_2d = tsne_input.fit_transform(x_input)
#                 x_centers_2d = tsne_input.fit_transform(x_centers)
#             else:
#                 x_2d = x_input
#                 x_centers_2d = x_centers

#             def plot_tsne(X_2d, labels, centers, title):
#                 scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.7, s=15)
#                 # plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='*', s=200, edgecolors='k', linewidths=1.5, label='Cluster Centers')
#                 plt.title(title)
#                 plt.xlabel("Dimension 1")
#                 plt.ylabel("Dimension 2")
#                 plt.colorbar(scatter, label='Class Labels')
#                 plt.grid(True)
#                 plt.axis('equal')
#             plt.figure(figsize=(6, 6))
#             plot_tsne(x_2d, y_pred, x_centers_2d, "IDEC")
#             plt.tight_layout()
#             plt.savefig(f"{vis_dir}/epoch_{epoch:03d}.jpg", dpi=300)

#             if epoch > 0 and delta_label < args.tol:
#                 print('delta_label {:.4f}'.format(delta_label), '< tol',
#                       args.tol)
#                 print('Reached tolerance threshold. Stopping training.')
#                 break

#         for batch_idx, (x, _, idx) in enumerate(train_loader):

#             x = x.to(device)
#             idx = idx.to(device)

#             x_bar, q = model(x)

#             reconstr_loss = F.mse_loss(x_bar, x)
#             kl_loss = F.kl_div(q.log(), p[idx])
#             loss = args.gamma * kl_loss + reconstr_loss

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()


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
            

            with open(log_path, "a") as f:
                log_str = 'Iter {:03d}: Acc {:.4f}, NMI {:.4f}, ARI {:.4f}, Δlabel {:.4f}\n'.format(
                    epoch, acc, nmi, ari, delta_label)
                print(log_str.strip())
                f.write(log_str)
            
            model.eval()
            cluster_centers = model.cluster_layer.data
            
            # 获取编码空间的表示和解码空间的表示
            with torch.no_grad():
                # 编码空间表示
                enc_h1 = F.relu(model.ae.enc_1(data))
                enc_h2 = F.relu(model.ae.enc_2(enc_h1))
                enc_h3 = F.relu(model.ae.enc_3(enc_h2))
                z_representation = model.ae.z_layer(enc_h3).cpu().numpy()
                
                # 解码空间表示（原始空间）
                dec_h1 = F.relu(model.ae.dec_1(cluster_centers))
                dec_h2 = F.relu(model.ae.dec_2(dec_h1))
                dec_h3 = F.relu(model.ae.dec_3(dec_h2))
                x_centers = model.ae.x_bar_layer(dec_h3).cpu().numpy()
                
                # 获取聚类中心在编码空间的表示
                enc_center_h1 = F.relu(model.ae.enc_1(cluster_centers))
                enc_center_h2 = F.relu(model.ae.enc_2(enc_center_h1))
                enc_center_h3 = F.relu(model.ae.enc_3(enc_center_h2))
                z_centers = model.ae.z_layer(enc_center_h3).cpu().numpy()
            
            # 原始空间数据
            x_input = dataset.x_row
            if x_input.shape[1] > 2:
                tsne_input = TSNE(n_components=2, random_state=42)
                x_2d = tsne_input.fit_transform(x_input)
                x_centers_2d = tsne_input.fit_transform(x_centers)
            else:
                x_2d = x_input
                x_centers_2d = x_centers
            
            # 编码空间数据（降维到2D）
            if z_representation.shape[1] > 2:
                tsne_z = TSNE(n_components=2, random_state=42)
                z_2d = tsne_z.fit_transform(z_representation)
                z_centers_2d = tsne_z.fit_transform(z_centers)
            else:
                z_2d = z_representation
                z_centers_2d = z_centers

            # 基础聚类可视化（不带中心点）
            def plot_basic_clustering_no_centers(X_2d, labels, title):
                """基础聚类可视化 - 不带中心点"""
                scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.7, s=15)
                plt.title(title)
                plt.xlabel("Dimension 1")
                plt.ylabel("Dimension 2")
                plt.colorbar(scatter, label='Class Labels')
                plt.grid(True)
                plt.axis('equal')

            # 基础聚类可视化（带中心点）
            def plot_basic_clustering_with_centers(X_2d, labels, centers, title):
                """基础聚类可视化 - 带中心点"""
                scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.7, s=15)
                plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='*', s=200, 
                           edgecolors='black', linewidths=1.5, label='Cluster Centers')
                plt.title(title)
                plt.xlabel("Dimension 1")
                plt.ylabel("Dimension 2")
                plt.colorbar(scatter, label='Class Labels')
                plt.grid(True)
                plt.axis('equal')
                plt.legend()

            # 维诺图可视化（使用您提供的风格）
            def plot_voronoi_with_boundaries(X_2d, labels, centers, title="Voronoi Diagram with Boundaries"):
                """绘制维诺图与决策边界的综合图"""
                # 检查是否有足够的点来创建维诺图
                if len(centers) < 3:
                    print(f"警告: 只有 {len(centers)} 个中心点，无法创建维诺图（至少需要3个点）")
                    # 绘制简单的决策边界
                    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap=plt.cm.Set1, alpha=0.7, s=50)
                    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=200, linewidths=3)
                    plt.title(f"{title} (Insufficient points for Voronoi - n={len(centers)})")
                    plt.xlabel("Dimension 1")
                    plt.ylabel("Dimension 2")
                    plt.grid(True, alpha=0.3)
                    return
                
                # 创建颜色映射
                unique_labels = np.unique(labels)
                colors = plt.cm.Set1(np.linspace(0, 1, max(3, len(unique_labels))))
                color_list = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow', 'lightpink', 
                              'lightcyan', 'wheat', 'lavender', 'lightgray', 'lightsteelblue']
                
                # 创建网格用于决策边界
                x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
                y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                     np.linspace(y_min, y_max, 100))
                grid_points = np.c_[xx.ravel(), yy.ravel()]
                
                # 计算每个网格点到中心的距离（维诺图分配）
                distances = np.sqrt(((grid_points[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2))
                voronoi_labels = np.argmin(distances, axis=1)
                
                # 绘制维诺图区域背景
                plt.scatter(grid_points[:, 0], grid_points[:, 1], 
                           c=[colors[label % len(colors)] for label in voronoi_labels], 
                           alpha=0.1, s=1)
                
                # 绘制原始数据点
                for i, label in enumerate(unique_labels):
                    cluster_points = X_2d[labels == label]
                    if len(cluster_points) > 0:
                        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                                   c=[colors[i % len(colors)]], alpha=0.8, s=60, 
                                   edgecolors='black', linewidth=0.5, label=f'Class {label}')
                
                # 绘制中心点
                plt.scatter(centers[:, 0], centers[:, 1], 
                           c='black', marker='x', s=300, linewidths=4, label='Centers')
                
                # 绘制决策边界（等高线）
                Z = voronoi_labels.reshape(xx.shape)
                plt.contour(xx, yy, Z, levels=np.arange(len(centers))-0.5, 
                            colors='black', linewidths=2, alpha=0.8)
                
                # 绘制维诺图边界
                try:
                    vor = Voronoi(centers)
                    voronoi_plot_2d(vor, show_vertices=False, line_colors='darkred', 
                                    line_width=2, line_style='--', point_size=0, ax=plt.gca())
                except Exception as e:
                    print(f"创建维诺图时出错: {e}")
                    print("继续绘制其他元素...")
                
                plt.title(title, fontsize=14)
                plt.xlabel("Dimension 1", fontsize=12)
                plt.ylabel("Dimension 2", fontsize=12)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xlim([x_min, x_max])
                plt.ylim([y_min, y_max])
                plt.grid(True, alpha=0.3)

            def save_idec_visualizations(X_2d_original, X_2d_encoded, labels, centers_original, centers_encoded, epoch, vis_dir):
                """为IDEC保存所有相关图片（原始空间和编码空间）"""
                epoch_dir = f"{vis_dir}/epoch_{epoch:03d}"
                os.makedirs(epoch_dir, exist_ok=True)
                
                # ==================== 原始空间可视化 ====================
                original_dir = f"{epoch_dir}/original_space"
                os.makedirs(original_dir, exist_ok=True)
                
                # 1. 原始空间 - 基础聚类结果（不带中心点）
                plt.figure(figsize=(8, 6))
                plot_basic_clustering_no_centers(X_2d_original, labels, f"Original Space - IDEC Clustering - Epoch {epoch} (No Centers)")
                plt.tight_layout()
                plt.savefig(f"{original_dir}/original_epoch_{epoch:03d}_clustering_no_centers.jpg", dpi=300, bbox_inches='tight')
                plt.close()
                
                # 2. 原始空间 - 基础聚类结果（带中心点）
                plt.figure(figsize=(8, 6))
                plot_basic_clustering_with_centers(X_2d_original, labels, centers_original, f"Original Space - IDEC Clustering - Epoch {epoch} (With Centers)")
                plt.tight_layout()
                plt.savefig(f"{original_dir}/original_epoch_{epoch:03d}_clustering_with_centers.jpg", dpi=300, bbox_inches='tight')
                plt.close()
                
                # 3. 原始空间 - 维诺图可视化
                plt.figure(figsize=(12, 10))
                plot_voronoi_with_boundaries(X_2d_original, labels, centers_original, f"Original Space - IDEC Voronoi Diagram - Epoch {epoch}")
                plt.tight_layout()
                plt.savefig(f"{original_dir}/original_epoch_{epoch:03d}_voronoi_diagram.jpg", dpi=300, bbox_inches='tight')
                plt.close()
                
                # ==================== 编码空间可视化 ====================
                encoded_dir = f"{epoch_dir}/encoded_space"
                os.makedirs(encoded_dir, exist_ok=True)
                
                # 4. 编码空间 - 基础聚类结果（不带中心点）
                plt.figure(figsize=(8, 6))
                plot_basic_clustering_no_centers(X_2d_encoded, labels, f"Encoded Space - IDEC Clustering - Epoch {epoch} (No Centers)")
                plt.tight_layout()
                plt.savefig(f"{encoded_dir}/encoded_epoch_{epoch:03d}_clustering_no_centers.jpg", dpi=300, bbox_inches='tight')
                plt.close()
                
                # 5. 编码空间 - 基础聚类结果（带中心点）
                plt.figure(figsize=(8, 6))
                plot_basic_clustering_with_centers(X_2d_encoded, labels, centers_encoded, f"Encoded Space - IDEC Clustering - Epoch {epoch} (With Centers)")
                plt.tight_layout()
                plt.savefig(f"{encoded_dir}/encoded_epoch_{epoch:03d}_clustering_with_centers.jpg", dpi=300, bbox_inches='tight')
                plt.close()
                
                # 6. 编码空间 - 维诺图可视化
                plt.figure(figsize=(12, 10))
                plot_voronoi_with_boundaries(X_2d_encoded, labels, centers_encoded, f"Encoded Space - IDEC Voronoi Diagram - Epoch {epoch}")
                plt.tight_layout()
                plt.savefig(f"{encoded_dir}/encoded_epoch_{epoch:03d}_voronoi_diagram.jpg", dpi=300, bbox_inches='tight')
                plt.close()
                
                # ==================== 对比可视化 ====================
                # 7. 原始空间 vs 编码空间对比图
                plt.figure(figsize=(15, 5))
                
                # 左侧：原始空间聚类（带中心）
                plt.subplot(1, 3, 1)
                scatter1 = plt.scatter(X_2d_original[:, 0], X_2d_original[:, 1], c=labels, cmap='viridis', alpha=0.7, s=15)
                plt.scatter(centers_original[:, 0], centers_original[:, 1], c='red', marker='*', s=200, 
                           edgecolors='black', linewidths=1.5)
                plt.title(f"Original Space - Epoch {epoch}")
                plt.xlabel("Dimension 1")
                plt.ylabel("Dimension 2")
                plt.grid(True)
                plt.axis('equal')
                plt.colorbar(scatter1, label='Class Labels')
                
                # 中间：编码空间聚类（带中心）
                plt.subplot(1, 3, 2)
                scatter2 = plt.scatter(X_2d_encoded[:, 0], X_2d_encoded[:, 1], c=labels, cmap='viridis', alpha=0.7, s=15)
                plt.scatter(centers_encoded[:, 0], centers_encoded[:, 1], c='red', marker='*', s=200, 
                           edgecolors='black', linewidths=1.5)
                plt.title(f"Encoded Space - Epoch {epoch}")
                plt.xlabel("Dimension 1")
                plt.ylabel("Dimension 2")
                plt.grid(True)
                plt.axis('equal')
                plt.colorbar(scatter2, label='Class Labels')
                
                # 右侧：维诺图对比
                plt.subplot(1, 3, 3)
                if len(centers_original) >= 3:
                    # 原始空间维诺图
                    x_min, x_max = X_2d_original[:, 0].min() - 1, X_2d_original[:, 0].max() + 1
                    y_min, y_max = X_2d_original[:, 1].min() - 1, X_2d_original[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                                       np.linspace(y_min, y_max, 50))
                    grid_points = np.c_[xx.ravel(), yy.ravel()]
                    distances = np.sqrt(((grid_points[:, np.newaxis, :] - centers_original[np.newaxis, :, :]) ** 2).sum(axis=2))
                    voronoi_labels = np.argmin(distances, axis=1)
                    plt.scatter(grid_points[:, 0], grid_points[:, 1], 
                               c=voronoi_labels, cmap=plt.cm.Set1, alpha=0.1, s=1)
                    scatter3 = plt.scatter(X_2d_original[:, 0], X_2d_original[:, 1], c=labels, cmap=plt.cm.Set1, alpha=0.7, s=15)
                    plt.scatter(centers_original[:, 0], centers_original[:, 1], c='black', marker='x', s=100, linewidths=2)
                    plt.contour(xx, yy, voronoi_labels.reshape(xx.shape), 
                               levels=np.arange(len(centers_original))-0.5, colors='black', linewidths=1)
                else:
                    scatter3 = plt.scatter(X_2d_original[:, 0], X_2d_original[:, 1], c=labels, cmap=plt.cm.Set1, alpha=0.7, s=15)
                    plt.scatter(centers_original[:, 0], centers_original[:, 1], c='black', marker='x', s=100, linewidths=2)
                
                plt.title(f"Original Space Voronoi - Epoch {epoch}")
                plt.xlabel("Dimension 1")
                plt.ylabel("Dimension 2")
                plt.grid(True)
                plt.axis('equal')
                
                plt.tight_layout()
                plt.savefig(f"{epoch_dir}/epoch_{epoch:03d}_space_comparison.jpg", dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"已为 epoch {epoch} 保存所有可视化结果到 {epoch_dir}/")
                print(f"  - 原始空间: {original_dir}/")
                print(f"  - 编码空间: {encoded_dir}/")

            # 保存可视化结果
            save_idec_visualizations(x_2d, z_2d, y_pred, x_centers_2d, z_centers_2d, epoch, vis_dir)

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


    print(args)
    train_idec()
