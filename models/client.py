"""
客户端本地LightGCN模型

核心设计：
1. 本地用户嵌入（不共享，保存在客户端）
2. 全局物品嵌入（从服务器接收，训练后返回更新）
3. BPR损失训练 + 本地邻域传播
"""
import torch
import torch.nn as nn
import numpy as np


class LocalLightGCN(nn.Module):
    """联邦场景下的客户端本地LightGCN"""
    def __init__(self, n_local_users, n_items, emb_dim=64, n_layers=3):
        super().__init__()
        self.n_local_users = n_local_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers

        # 本地用户嵌入（不共享）
        self.local_user_embedding = nn.Embedding(n_local_users, emb_dim)
        nn.init.normal_(self.local_user_embedding.weight, std=0.1)

    def forward(self, local_adj, item_emb=None):
        """前向传播：在本地子图上做LightGCN传播
        Args:
            local_adj: 本地邻接矩阵（稀疏张量）
            item_emb: 物品嵌入（None时用self.global_item_emb）
        Returns:
            all_embeddings: 拼接的[user_emb, item_emb]
        """
        if item_emb is None:
            item_emb = self.global_item_emb
        device = self.local_user_embedding.weight.device
        if item_emb.device != device:
            item_emb = item_emb.to(device)

        ego_embeddings = torch.cat([
            self.local_user_embedding.weight,
            item_emb
        ], dim=0)

        all_embeddings = [ego_embeddings]
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(local_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        return torch.mean(torch.stack(all_embeddings), dim=0)

    def bpr_loss(self, users_local, pos_items, neg_items, local_adj, item_emb=None):
        """BPR损失 + L2正则（本地数据集）
        Args:
            users_local: 本地用户ID（0..n_local_users-1）
            pos_items: 正例物品ID（全局物品ID）
            neg_items: 负例物品ID（全局物品ID）
            local_adj: 本地邻接矩阵
            item_emb: 物品嵌入（None时用self.global_item_emb）
        """
        all_embeddings = self.forward(local_adj, item_emb=item_emb)
        user_emb = all_embeddings[users_local]
        pos_emb = all_embeddings[self.n_local_users + pos_items]
        neg_emb = all_embeddings[self.n_local_users + neg_items]

        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))

        reg = (1.0 / 2) * (
            user_emb.norm(2).pow(2) +
            pos_emb.norm(2).pow(2) +
            neg_emb.norm(2).pow(2)
        ) / users_local.shape[0]

        return loss + reg

    def get_item_embedding_update(self, global_item_emb_before):
        """计算物品嵌入更新量 delta = local - global
        Args:
            global_item_emb_before: 训练前的全局物品嵌入
        Returns:
            delta: 物品嵌入更新量
        """
        return self.global_item_emb.detach() - global_item_emb_before

    def load_global_item_emb(self, global_item_emb):
        """接收服务器下发的全局物品嵌入"""
        self.global_item_emb = global_item_emb.detach().to(
            self.local_user_embedding.weight.device
        )
