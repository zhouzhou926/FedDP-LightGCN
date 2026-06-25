"""
配置加载模块 - 读取fed_config.yaml并解析为FedConfig对象
"""
import os
import yaml
import torch
import numpy as np
import random


class FedConfig:
    """联邦训练配置"""
    def __init__(self, config_path="./config/fed_config.yaml"):
        self.n_clients = 10
        self.partition = "uniform"
        self.dirichlet_alpha = 0.5
        self.seed = 2024
        self.local_epochs = 5
        self.batch_size = 1024
        self.lr = 0.001
        self.decay = 0.0001
        self.emb_dim = 64
        self.n_layers = 3
        self.fed_rounds = 50
        self.client_frac = 1.0
        self.enable_dp = False
        self.epsilon = 10.0
        self.delta = 1.0e-05
        self.budget_strategy = "adaptive"
        self.clip_strategy = "data_aware"
        self.sparse_dp = True
        self.eval_interval = 5
        self.top_k = [20]
        self.data_path = "./data/gowalla"
        self.dataset = ""

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if cfg:
                for k, v in cfg.items():
                    if hasattr(self, k):
                        setattr(self, k, v)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._set_seed(self.seed)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_dataset_path(self):
        if self.dataset:
            return os.path.join(self.data_path, self.dataset)
        return self.data_path

    def __repr__(self):
        return (
            f"FedConfig(clients={self.n_clients}, partition={self.partition}, "
            f"rounds={self.fed_rounds}, dp={self.enable_dp}, "
            f"eps={self.epsilon}, emb_dim={self.emb_dim})"
        )
