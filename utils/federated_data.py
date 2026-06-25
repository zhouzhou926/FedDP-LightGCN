"""
联邦数据划分模块 - 将用户分配到各客户端，构建客户端本地子图
"""
import numpy as np
from collections import defaultdict


class FederatedDataPartitioner:
    """联邦数据划分器：将用户分配到K个客户端"""
    def __init__(self, dataset, n_clients=10, partition="uniform",
                 dirichlet_alpha=0.5, seed=2024):
        self.dataset = dataset
        self.n_clients = n_clients
        self.partition = partition
        self.dirichlet_alpha = dirichlet_alpha
        np.random.seed(seed)
        self.client_data = self._partition()

    def _partition(self):
        n_users = self.dataset.n_users
        if self.partition == "uniform":
            users_per_client = n_users // self.n_clients
            indices = np.random.permutation(n_users)
            client_users = {}
            for c in range(self.n_clients):
                start = c * users_per_client
                end = start + users_per_client if c < self.n_clients - 1 else n_users
                client_users[c] = sorted(indices[start:end].tolist())
        elif self.partition == "dirichlet":
            from numpy.random import dirichlet
            alpha = [self.dirichlet_alpha] * self.n_clients
            proportions = dirichlet(alpha, size=n_users)
            assignments = np.argmax(proportions, axis=1)
            client_users = defaultdict(list)
            for u, c in enumerate(assignments):
                client_users[int(c)].append(u)
            for c in client_users:
                client_users[c] = sorted(client_users[c])
        else:
            raise ValueError(f"Unknown partition: {self.partition}")

        client_data = {}
        user_pos_dict = {}
        for u, items in self.dataset.train_data:
            user_pos_dict[u] = set(items)

        for c, users in client_users.items():
            user_set = set(users)
            local_train = [(u, items) for u, items in self.dataset.train_data if u in user_set]
            local_test = [(u, items) for u, items in self.dataset.test_data if u in user_set]

            global_to_local = {gid: lid for lid, gid in enumerate(sorted(users))}
            local_to_global = {lid: gid for gid, lid in global_to_local.items()}

            n_train_inter = sum(len(items) for _, items in local_train)

            client_data[c] = {
                "users": users,
                "n_users": len(users),
                "global_to_local": global_to_local,
                "local_to_global": local_to_global,
                "train_data": local_train,
                "test_data": local_test,
                "n_train_inter": n_train_inter,
                "user_pos_dict": user_pos_dict,
            }
        return client_data

    def build_local_subgraph(self, client_id):
        """为指定客户端构建本地LightGCN子图"""
        data = self.client_data[client_id]
        n_local = data["n_users"]
        n_items = self.dataset.n_items
        n_nodes = n_local + n_items
        global_to_local = data["global_to_local"]

        rows, cols = [], []
        for user, items in data["train_data"]:
            local_u = global_to_local[user]
            for item in items:
                item_idx = n_local + item
                rows.append(local_u)
                cols.append(item_idx)
                rows.append(item_idx)
                cols.append(local_u)

        if len(rows) == 0:
            return None

        values = np.ones(len(rows))
        import scipy.sparse as sp
        adj = sp.coo_matrix((values, (rows, cols)), shape=(n_nodes, n_nodes))
        row_sum = np.array(adj.sum(axis=1)).flatten()
        d_inv_sqrt = np.zeros_like(row_sum)
        mask = row_sum > 0
        d_inv_sqrt[mask] = np.power(row_sum[mask], -0.5)
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        return norm_adj.tocoo()

    def get_local_adj_tensor(self, client_id, device):
        """获取客户端的本地邻接矩阵（PyTorch稀疏张量）"""
        import torch
        adj = self.build_local_subgraph(client_id)
        if adj is None:
            return None
        indices_np = np.array([adj.row, adj.col])
        indices = torch.tensor(indices_np, dtype=torch.long)
        values = torch.tensor(adj.data, dtype=torch.float32)
        shape = torch.Size(adj.shape)
        return torch.sparse_coo_tensor(indices, values, shape).to(device)

    def get_client_info(self):
        info = {}
        for c, data in self.client_data.items():
            info[c] = {
                "n_users": data["n_users"],
                "n_train": data["n_train_inter"],
                "avg_inter": data["n_train_inter"] / max(data["n_users"], 1),
            }
        return info

    def print_summary(self):
        info = self.get_client_info()
        n_users = sum(v["n_users"] for v in info.values())
        n_train = sum(v["n_train"] for v in info.values())
        print(f"Clients: {self.n_clients} | Total users: {n_users} | Total inter: {n_train}")
        print(f"Partition: {self.partition}")
        for c, v in info.items():
            print(f"  Client {c:2d}: users={v['n_users']:4d}, inter={v['n_train']:6d}, avg={v['avg_inter']:.1f}")
