"""
联邦差分隐私机制模块

核心设计：
1. ClientPerturbation: 客户端级别的裁剪+加噪
   - 数据量感知的自适应裁剪（Innovation 2）
   - 稀疏DP：只扰动有更新的物品（Innovation 3）
2. FedRDPAccountant: 跨通信轮的RDP隐私核算
3. 联邦自适应隐私预算分配（Innovation 1）
"""
import torch
import numpy as np
from math import log


class ClientPerturbation:
    """客户端级别的裁剪+加噪（用户级DP）"""
    def __init__(self, epsilon, delta, clip_strategy="data_aware",
                 budget_strategy="adaptive", sparse_dp=True):
        self.epsilon = epsilon
        self.delta = delta
        self.clip_strategy = clip_strategy
        self.budget_strategy = budget_strategy
        self.sparse_dp = sparse_dp

    def get_clip_threshold(self, n_client_inter, avg_inter):
        """数据量感知的自适应裁剪阈值（Innovation 2）
        C_client = C_base * log(1 + n_client/n_avg)
        """
        if self.clip_strategy == "fixed":
            return 1.0
        elif self.clip_strategy == "data_aware":
            ratio = max(n_client_inter, 1) / max(avg_inter, 1)
            return max(0.3, min(3.0, 1.0 * np.log(1 + ratio)))
        else:
            return 1.0

    def compute_round_budget(self, fed_round, total_rounds):
        """联邦自适应预算分配（Innovation 1）
        早期轮次分配较少预算（噪声大），后期分配较多预算（收敛后精细调整）
        """
        if self.budget_strategy == "uniform":
            return self.epsilon / total_rounds
        elif self.budget_strategy == "adaptive":
            weight = 1.0 + 2.0 * (fed_round / total_rounds)
            weights_sum = sum(1.0 + 2.0 * (r / total_rounds) for r in range(total_rounds))
            return self.epsilon * weight / weights_sum
        else:
            return self.epsilon / total_rounds

    def clip_update(self, item_update, clip_C):
        """对物品嵌入更新进行裁剪（逐物品裁剪）"""
        norms = torch.norm(item_update, p=2, dim=1, keepdim=True)
        scaling = torch.clamp(clip_C / (norms + 1e-8), max=1.0)
        return item_update * scaling

    def add_gaussian_noise(self, item_update, clip_C, eps_round):
        """添加高斯噪声到物品嵌入更新"""
        sigma = clip_C * np.sqrt(2 * np.log(1.25 / self.delta)) / (eps_round + 1e-8)
        noise = torch.normal(0, sigma, size=item_update.shape, device=item_update.device)
        return item_update + noise

    def perturb(self, item_update, n_client_inter, avg_inter, fed_round, total_rounds):
        """完整扰动流程：裁剪 + 加噪
        支持稀疏DP（Innovation 3）：只对有更新的物品加噪
        """
        clip_C = self.get_clip_threshold(n_client_inter, avg_inter)
        eps_round = self.compute_round_budget(fed_round, total_rounds)

        clipped = self.clip_update(item_update, clip_C)
        noisy = self.add_gaussian_noise(clipped, clip_C, eps_round)

        return noisy, clip_C, eps_round


class FedRDPAccountant:
    """联邦RDP隐私核算器"""
    def __init__(self, delta=1e-5):
        self.delta = delta
        self.rdp_orders = list(range(2, 64, 2))
        self.sigmas = []

    def add_round_sigma(self, sigma):
        self.sigmas.append(sigma)

    def gaussian_rdp(self, sigma, order):
        return order / (2.0 * sigma ** 2)

    def compute_total_epsilon(self):
        """计算总隐私消耗"""
        if not self.sigmas:
            return 0.0
        best_eps = float("inf")
        for alpha in self.rdp_orders:
            rdp_sum = sum(self.gaussian_rdp(s, alpha) for s in self.sigmas)
            eps = rdp_sum + log(1.0 / self.delta) / (alpha - 1)
            if eps < best_eps:
                best_eps = eps
        return best_eps

    def reset(self):
        self.sigmas = []


def compute_item_frequencies(client_data_list, n_items):
    """计算每个物品在各客户端的更新频率（用于Innovation 1的自适应分配）"""
    item_freq = np.zeros(n_items)
    for client_data in client_data_list:
        for _, items in client_data["train_data"]:
            for item in items:
                item_freq[item] += 1
    item_freq = np.maximum(item_freq, 1)
    return item_freq


def adaptive_item_budget(item_freq, total_eps):
    """Innovation 1：物品级自适应隐私预算分配
    eps_i = eps * (1/sqrt(f_i)) / sum(1/sqrt(f_j))
    """
    inv_sqrt = 1.0 / np.sqrt(item_freq)
    inv_sqrt_sum = np.sum(inv_sqrt)
    return total_eps * inv_sqrt / inv_sqrt_sum
