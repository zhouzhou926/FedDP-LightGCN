"""
联邦服务端模型 - FedAvg聚合 + 全局物品嵌入管理 + DP核算

核心职责：
1. 维护全局物品嵌入
2. FedAvg聚合客户端更新
3. 隐私核算与追踪
4. 支持加权聚合（按客户端数据量）
"""
import torch
import torch.nn as nn
import numpy as np
from models.dp_mechanism import FedRDPAccountant


class FedServer(nn.Module):
    """联邦服务端"""
    def __init__(self, n_items, emb_dim=64, config=None):
        super().__init__()
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.config = config

        # 全局物品嵌入
        self.global_item_embedding = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.global_item_embedding.weight, std=0.1)

        # RDP隐私核算
        self.accountant = FedRDPAccountant(delta=config.delta if config else 1e-5)

        # 追踪聚合状态
        self.round_num = 0
        self.epsilon_consumed = 0.0

    def distribute(self, device=None):
        """下发全局物品嵌入到客户端"""
        return self.global_item_embedding.weight.detach().clone()

    def aggregate(self, client_updates, client_weights, fed_round, apply_dp=False):
        """FedAvg聚合客户端更新
        Args:
            client_updates: list of (item_emb_delta, clip_C, eps_round) tuples
            client_weights: list of client weights (data volume proportion)
            fed_round: 当前联邦轮次
            apply_dp: 是否在服务端加噪（备用，主要用于客户端DP场景）
        Returns:
            aggregated_delta: 聚合后的更新量
        """
        self.round_num = fed_round

        # 加权平均
        total_weight = sum(client_weights)
        if total_weight <= 0:
            return None

        weighted_sum = torch.zeros_like(self.global_item_embedding.weight)
        for i, (delta, _, _) in enumerate(client_updates):
            weighted_sum += delta.to(weighted_sum.device) * (client_weights[i] / total_weight)

        # 应用更新
        with torch.no_grad():
            self.global_item_embedding.weight.add_(weighted_sum)

        # 记录噪声信息
        sigmas = [sigma for _, _, sigma in client_updates if sigma > 0]
        for s in sigmas:
            self.accountant.add_round_sigma(s)

        return weighted_sum

    def compute_epsilon(self):
        """计算当前总隐私消耗"""
        self.epsilon_consumed = self.accountant.compute_total_epsilon()
        return self.epsilon_consumed

    def get_state_dict(self):
        """返回服务端状态（用于保存）"""
        return {
            "round": self.round_num,
            "item_emb": self.global_item_embedding.weight.detach().cpu().numpy(),
            "epsilon_consumed": self.epsilon_consumed,
            "sigmas": self.accountant.sigmas,
        }

    def load_state(self, state):
        """加载服务端状态"""
        self.round_num = state.get("round", 0)
        self.global_item_embedding.weight.data = torch.tensor(
            state["item_emb"]
        )
        self.epsilon_consumed = state.get("epsilon_consumed", 0.0)
        self.accountant.sigmas = state.get("sigmas", [])
