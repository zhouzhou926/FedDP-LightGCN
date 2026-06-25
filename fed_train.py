"""
FedDP-LightGCN: 联邦差分隐私LightGCN推荐系统主训练脚本

用法:
  python fed_train.py                                   # 使用默认配置
  python fed_train.py --config ./config/fed_config.yaml  # 指定配置
  python fed_train.py --dp --epsilon 10.0                # 启用DP

核心流程（每轮联邦训练）：
1. 服务器下发全局物品嵌入到各客户端
2. 各客户端本地训练LightGCN（本地用户嵌入 + 全局物品嵌入）
3. 客户端对物品嵌入更新进行裁剪+加噪（DP）
4. 服务器FedAvg聚合各客户端的加噪更新
5. 定期评估所有客户端的本地指标
"""
import os
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from utils.config import FedConfig
from utils.data_loader import LightGCNDataset, BatchSampler
from utils.federated_data import FederatedDataPartitioner
from utils.metrics import federated_evaluate, print_metrics
from models.client import LocalLightGCN
from models.server import FedServer
from models.dp_mechanism import ClientPerturbation

class FedDPTrainer:
    """FedDP-LightGCN训练器"""
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.results_log = []

        print(f"[Config] {config}")
        print(f"[Device] {self.device}")

    def load_data(self):
        """加载数据集"""
        data_path = self.config.get_dataset_path()
        print(f"[Data] Loading dataset from {data_path}")
        self.dataset = LightGCNDataset(data_path)
        print(f"[Data] Users={self.dataset.n_users}, Items={self.dataset.n_items}")
        print(f"[Data] Train pairs={sum(len(i) for _,i in self.dataset.train_data)}")
        return self.dataset

    def partition_data(self):
        """划分联邦客户端数据"""
        print(f"[Fed] Partitioning {self.dataset.n_users} users into {self.config.n_clients} clients")
        self.partitioner = FederatedDataPartitioner(
            dataset=self.dataset,
            n_clients=self.config.n_clients,
            partition=self.config.partition,
            dirichlet_alpha=self.config.dirichlet_alpha,
            seed=self.config.seed,
        )
        self.partitioner.print_summary()
        return self.partitioner

    def initialize_models(self):
        """初始化服务端和客户端模型 + 缓存邻接矩阵 + 用户映射"""
        n_items = self.dataset.n_items
        emb_dim = self.config.emb_dim
        n_layers = self.config.n_layers

        # 服务端
        self.server = FedServer(
            n_items=n_items,
            emb_dim=emb_dim,
            config=self.config,
        ).to(self.device)

        # 客户端本地模型
        self.clients = {}
        for c in range(self.config.n_clients):
            n_local = self.partitioner.client_data[c]["n_users"]
            self.clients[c] = LocalLightGCN(
                n_local_users=n_local,
                n_items=n_items,
                emb_dim=emb_dim,
                n_layers=n_layers,
            ).to(self.device)

        # === 缓存邻接矩阵（避免每轮重建） ===
        self.cached_adjs = {}
        n_active = 0
        for c in range(self.config.n_clients):
            adj = self.partitioner.get_local_adj_tensor(c, self.device)
            if adj is not None:
                self.cached_adjs[c] = adj
                n_active += 1
        print(f"[Cache] Cached {n_active}/{self.config.n_clients} adjacency matrices on {self.device}")

        # === 预计算用户ID映射张量（避免逐batch的Python dict lookup） ===
        self.user_id_maps = {}
        for c in range(self.config.n_clients):
            g2l = self.partitioner.client_data[c]["global_to_local"]
            if not g2l:
                continue
            max_uid = max(g2l.keys())
            mapping = torch.zeros(max_uid + 1, dtype=torch.long)
            for gid, lid in g2l.items():
                mapping[gid] = lid
            self.user_id_maps[c] = mapping.to(self.device)
        print(f"[Cache] Pre-computed {len(self.user_id_maps)} user ID mapping tensors")

        # DP扰动器（仅客户端DP）
        if self.config.enable_dp:
            self.perturbator = ClientPerturbation(
                epsilon=self.config.epsilon,
                delta=self.config.delta,
                clip_strategy=self.config.clip_strategy,
                budget_strategy=self.config.budget_strategy,
                sparse_dp=self.config.sparse_dp,
            )
            print(f"[DP] epsilon={self.config.epsilon}, delta={self.config.delta}")
            print(f"[DP] clip={self.config.clip_strategy}, budget={self.config.budget_strategy}")
            print(f"[DP] sparse_dp={self.config.sparse_dp}")

    def compute_client_weights(self):
        """计算各客户端的聚合权重（按数据量）"""
        total_inter = sum(
            self.partitioner.client_data[c]["n_train_inter"]
            for c in range(self.config.n_clients)
        )
        weights = []
        for c in range(self.config.n_clients):
            n_inter = self.partitioner.client_data[c]["n_train_inter"]
            weights.append(n_inter / total_inter if total_inter > 0 else 1.0 / self.config.n_clients)
        return weights

    def client_local_train(self, client_id, global_item_emb, fed_round):
        """单个客户端的本地训练"""
        client = self.clients[client_id]
        client_data = self.partitioner.client_data[client_id]
        # 使用缓存的邻接矩阵
        local_adj = self.cached_adjs.get(client_id)

        if local_adj is None:
            return None

        # 加载全局物品嵌入
        global_item_emb_before = global_item_emb.detach().clone()

        # === 修复：创建可训练的物品嵌入副本 ===
        local_item_emb = nn.Parameter(
            global_item_emb.detach().clone().to(self.device)
        )

        # 优化器同时更新用户嵌入 + 物品嵌入
        optimizer = optim.Adam(
            list(client.parameters()) + [local_item_emb],
            lr=self.config.lr,
            weight_decay=self.config.decay,
        )

        local_epochs = self.config.local_epochs
        batch_size = min(max(2048, self.config.batch_size), client_data["n_train_inter"])

        sampler = BatchSampler(
            n_users=client_data["n_users"],
            n_items=self.dataset.n_items,
            train_data=client_data["train_data"],
        )

        for epoch in range(local_epochs):
            client.train()
            n_batches = max(1, client_data["n_train_inter"] // batch_size)
            total_loss = 0.0
            start_t = time.time()

            local_adj = self.cached_adjs.get(client_id)

            for batch_idx in range(n_batches):
                users, pos_items, neg_items = sampler.sample_batch(batch_size)

                users_local = self.user_id_maps[client_id][users].to(self.device)
                pos_items = pos_items.to(self.device)
                neg_items = neg_items.to(self.device)

                # 关键：传入可训练物品嵌入
                loss = client.bpr_loss(users_local, pos_items, neg_items, local_adj, item_emb=local_item_emb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - start_t
                avg_loss = total_loss / (batch_idx + 1)
                print(f"    [Client {client_id}] epoch {epoch+1}/{local_epochs}, batch {batch_idx + 1}/{n_batches}, loss={avg_loss:.4f}, {elapsed:.1f}s", flush=True)

        # 计算物品嵌入更新量
        item_delta = local_item_emb.detach().cpu() - global_item_emb.cpu()


        # DP扰动
        clip_C = 0.0
        eps_round = 0.0
        if self.config.enable_dp:
            avg_inter = np.mean([
                v["avg_inter"] for v in self.partitioner.get_client_info().values()
            ])
            item_delta, clip_C, eps_round = self.perturbator.perturb(
                item_delta,
                n_client_inter=client_data["n_train_inter"],
                avg_inter=avg_inter,
                fed_round=fed_round,
                total_rounds=self.config.fed_rounds,
            )

        return item_delta.detach().cpu(), clip_C, eps_round

    def train(self):
        """联邦训练主循环"""
        client_weights = self.compute_client_weights()
        print(f"[Train] Starting federated training for {self.config.fed_rounds} rounds")
        print(f"[Train] Clients: {self.config.n_clients}, local_epochs: {self.config.local_epochs}")
        print(f"[Train] Client weights computed (min={min(client_weights):.4f}, max={max(client_weights):.4f})")

        start_time = time.time()

        for fed_round in range(self.config.fed_rounds):
            round_start = time.time()

            # 服务器下发全局物品嵌入
            global_item_emb = self.server.distribute().to(self.device)

            # 客户端本地训练 + DP扰动
            client_updates = []
            skip_clients = 0
            for c in range(self.config.n_clients):
                update = self.client_local_train(c, global_item_emb, fed_round)
                if update is None:
                    skip_clients += 1
                    continue
                client_updates.append(update)

            # 服务端聚合
            if len(client_updates) == 0:
                print(f"[Round {fed_round:3d}] WARNING: No client updates, skipping")
                continue

            # 取参与聚合的客户端对应的权重（用缓存的adj判定）
            active_weights = []
            for c in range(self.config.n_clients):
                if c in self.cached_adjs:
                    active_weights.append(client_weights[c])
            if len(active_weights) != len(client_updates):
                active_weights = [1.0 / len(client_updates)] * len(client_updates)

            with torch.no_grad():
                for upd in client_updates:
                    delta = upd[0]
                    dn = delta.norm(2, dim=1).mean().item()

            self.server.aggregate(
                client_updates, active_weights, fed_round,
                apply_dp=self.config.enable_dp,
            )

            # 评估
            round_time = time.time() - round_start
            eval_str = ""
            if ((fed_round + 1) % self.config.eval_interval == 0 or
                    fed_round == self.config.fed_rounds - 1):

                # 将聚合后的全局物品嵌入下发到各客户端做评估
                new_global_emb = self.server.distribute()
                for c in range(self.config.n_clients):
                    self.clients[c].load_global_item_emb(new_global_emb)

                results = self.evaluate(f"Round {fed_round + 1}")
                eval_str = f" | R@20={results.get('Recall@20', 0):.4f} N@20={results.get('NDCG@20', 0):.4f}"

                # 记录结果
                self.results_log.append({
                    "round": fed_round + 1,
                    "recall_20": results.get("Recall@20", 0),
                    "ndcg_20": results.get("NDCG@20", 0),
                    "epsilon": self.server.compute_epsilon() if self.config.enable_dp else 0,
                })

            eps_str = ""
            if self.config.enable_dp:
                eps_str = f" eps={self.server.compute_epsilon():.2f}"

            print(f"[Round {fed_round+1:3d}/{self.config.fed_rounds}] "
                  f"time={round_time:.1f}s clients={len(client_updates)}{eps_str}{eval_str}")

        total_time = time.time() - start_time
        print(f"\n[Train] Done! Total time: {total_time:.1f}s")

        # 打印最终结果
        if self.results_log:
            final = self.results_log[-1]
            print(f"[Result] Final Recall@20 = {final['recall_20']:.6f}")
            print(f"[Result] Final NDCG@20   = {final['ndcg_20']:.6f}")
            if self.config.enable_dp:
                print(f"[Result] Epsilon consumed = {final['epsilon']:.4f}")

        return self.results_log

    def evaluate(self, tag=""):
        """评估所有客户端（使用缓存的邻接矩阵）"""
        results = federated_evaluate(
            clients=self.clients,
            local_models=self.clients,
            partitioner=self.partitioner,
            device=self.device,
            top_k_list=self.config.top_k,
            cached_adjs=self.cached_adjs,
        )
        if tag:
            print(f"[Eval] {tag}")
        print_metrics(results)
        return results

def main():
    parser = argparse.ArgumentParser(description="FedDP-LightGCN")
    parser.add_argument("--config", type=str, default="./config/fed_config.yaml",
                        help="Config file path")
    parser.add_argument("--dp", action="store_true", help="Enable DP")
    parser.add_argument("--epsilon", type=float, default=None, help="DP epsilon")
    parser.add_argument("--clients", type=int, default=None, help="Number of clients")
    parser.add_argument("--rounds", type=int, default=None, help="Federated rounds")
    parser.add_argument("--local_epochs", type=int, default=None, help="Local training epochs")
    parser.add_argument("--data", type=str, default=None, help="Dataset path")
    parser.add_argument("--emb_dim", type=int, default=None, help="Embedding dimension")
    parser.add_argument("--no_dp", dest="dp", action="store_false", help="Disable DP")
    parser.set_defaults(dp=None)
    args = parser.parse_args()

    # 加载配置
    config = FedConfig(config_path=args.config)

    # 命令行参数覆盖
    if args.dp is not None:
        config.enable_dp = args.dp
    if args.epsilon is not None:
        config.epsilon = args.epsilon
    if args.clients is not None:
        config.n_clients = args.clients
    if args.rounds is not None:
        config.fed_rounds = args.rounds
    if args.local_epochs is not None:
        config.local_epochs = args.local_epochs
    if args.data is not None:
        config.data_path = args.data
    if args.emb_dim is not None:
        config.emb_dim = args.emb_dim

    # 训练
    trainer = FedDPTrainer(config)
    trainer.load_data()
    trainer.partition_data()
    trainer.initialize_models()
    trainer.train()

    # 打印摘要
    print("\n" + "=" * 50)
    print("Training Summary:")
    print(f"  Config: {config}")
    print(f"  Results: {len(trainer.results_log)} evaluations logged")
    if trainer.results_log:
        final = trainer.results_log[-1]
        print(f"  Final Recall@20 = {final['recall_20']:.6f}")
        print(f"  Final NDCG@20   = {final['ndcg_20']:.6f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
