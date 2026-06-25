"""
数据加载模块 - 兼容DP-LightGCN的数据格式
"""
import os
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

torch.sparse.check_sparse_tensor_invariants.enable()


class LightGCNDataset(Dataset):
    """与DP-LightGCN相同的LightGCN数据集加载器"""
    def __init__(self, data_path):
        self.data_path = data_path
        self.train_data, self.test_data = self._load_data()
        self.n_users, self.n_items = self._get_statics()
        self.adj_matrix = self._build_adj_matrix()
        self.test_data_idx = {}
        for user, items in self.test_data:
            self.test_data_idx[user] = (set(items), len(items))

    def _load_data(self):
        train_data = []
        test_data = []
        with open(os.path.join(self.data_path, "train.txt"), "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                user = int(parts[0])
                items = [int(x) for x in parts[1:]]
                train_data.append((user, items))
        with open(os.path.join(self.data_path, "test.txt"), "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                user = int(parts[0])
                items = [int(x) for x in parts[1:]]
                test_data.append((user, items))
        return train_data, test_data

    def _get_statics(self):
        all_users = [u for u, _ in self.train_data] + [u for u, _ in self.test_data]
        all_items = (
            [max(items) for _, items in self.train_data if items] +
            [max(items) for _, items in self.test_data if items]
        )
        n_users = max(all_users) + 1 if all_users else 0
        n_items = max(all_items) + 1 if all_items else 0
        return n_users, n_items

    def _build_adj_matrix(self):
        n_nodes = self.n_users + self.n_items
        rows, cols = [], []
        for user, items in self.train_data:
            for item in items:
                rows.append(user)
                cols.append(self.n_users + item)
                rows.append(self.n_users + item)
                cols.append(user)
        data = np.ones(len(rows))
        adj = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        row_sum = np.array(adj.sum(axis=1)).flatten()
        d_inv_sqrt = np.zeros_like(row_sum)
        mask = row_sum > 0
        d_inv_sqrt[mask] = np.power(row_sum[mask], -0.5)
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        return norm_adj.tocoo()

    def get_adj_tensor(self, device):
        indices_np = np.array([self.adj_matrix.row, self.adj_matrix.col])
        indices = torch.tensor(indices_np, dtype=torch.long)
        values = torch.tensor(self.adj_matrix.data, dtype=torch.float32)
        shape = torch.Size(self.adj_matrix.shape)
        return torch.sparse_coo_tensor(indices, values, shape).to(device)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        user, pos_items = self.train_data[idx]
        return user, pos_items


class BatchSampler:
    """BPR采样器"""
    def __init__(self, n_users, n_items, train_data):
        self.n_users = n_users
        self.n_items = n_items
        self.users = [u for u, _ in train_data]
        self.user_pos_dict = {}
        for u, items in train_data:
            self.user_pos_dict[u] = set(items)

    def sample_batch(self, batch_size):
        batch_users = np.random.choice(self.users, batch_size, replace=True)
        pos_items = []
        neg_items = []
        for u in batch_users:
            pos = np.random.choice(list(self.user_pos_dict[u]))
            pos_items.append(pos)
            neg = np.random.randint(0, self.n_items)
            while neg in self.user_pos_dict[u]:
                neg = np.random.randint(0, self.n_items)
            neg_items.append(neg)
        return (
            torch.LongTensor(batch_users),
            torch.LongTensor(pos_items),
            torch.LongTensor(neg_items)
        )
