"""
评估指标模块 - 联邦场景下的Recall@K, NDCG@K
"""
import numpy as np
import torch


def local_evaluate(model, client_data, local_adj, device, top_k_list=[20]):
    """在单个客户端上评估推荐性能"""
    model.eval()
    with torch.no_grad():
        all_embeddings = model(local_adj)
        n_local = client_data["n_users"]
        n_items = model.n_items
        u_emb = all_embeddings[:n_local]
        i_emb = all_embeddings[n_local:]

        recall_list = {k: [] for k in top_k_list}
        ndcg_list = {k: [] for k in top_k_list}

        local_to_global = client_data["local_to_global"]
        user_pos_dict = client_data["user_pos_dict"]
        test_data = client_data["test_data"]
        global_to_local = client_data["global_to_local"]

        test_users_set = set(u for u, _ in test_data)
        local_user_ids = sorted(global_to_local.values())

        batch_size = 256
        for start in range(0, len(local_user_ids), batch_size):
            end = min(start + batch_size, len(local_user_ids))
            batch_local_ids = local_user_ids[start:end]

            scores_gpu = torch.matmul(u_emb[batch_local_ids], i_emb.t())
            scores_np = scores_gpu.cpu().numpy()

            for local_idx, local_uid in enumerate(batch_local_ids):
                global_uid = local_to_global[local_uid]
                if global_uid not in test_users_set:
                    continue

                test_items = None
                n_test = 0
                for u, items in test_data:
                    if u == global_uid:
                        test_items = set(items)
                        n_test = len(items)
                        break
                if test_items is None or n_test == 0:
                    continue

                scores = scores_np[local_idx].copy()
                train_items = user_pos_dict.get(global_uid, set())
                mask_items = [item for item in train_items if item not in test_items]
                if mask_items:
                    scores[mask_items] = -1e9

                max_k = max(top_k_list)
                topk_idx = np.argsort(scores)[-max_k:][::-1]

                for k in top_k_list:
                    top_k_set = set(topk_idx[:k])
                    hits = top_k_set & test_items
                    n_hits = len(hits)
                    recall_list[k].append(n_hits / n_test)

                    dcg = sum(
                        1.0 / np.log2(rank + 2)
                        for rank, item in enumerate(topk_idx[:k])
                        if item in test_items
                    )
                    idcg = sum(
                        1.0 / np.log2(i + 2) for i in range(min(n_test, k))
                    )
                    ndcg_list[k].append(dcg / idcg if idcg > 0 else 0.0)

        results = {}
        for k in top_k_list:
            results[f"Recall@{k}"] = float(np.mean(recall_list[k])) if recall_list[k] else 0.0
            results[f"NDCG@{k}"] = float(np.mean(ndcg_list[k])) if ndcg_list[k] else 0.0
        return results


def federated_evaluate(clients, local_models, partitioner, device, top_k_list=[20], cached_adjs=None):
    """所有客户端加权平均评估（使用缓存的邻接矩阵）"""
    all_recall = {k: [] for k in top_k_list}
    all_ndcg = {k: [] for k in top_k_list}
    total_inter = sum(
        partitioner.client_data[c]["n_train_inter"]
        for c in range(partitioner.n_clients)
    )

    for c in range(partitioner.n_clients):
        client_data = partitioner.client_data[c]
        # 使用缓存的邻接矩阵（避免重复构建）
        if cached_adjs is not None and c in cached_adjs:
            local_adj = cached_adjs[c]
        else:
            local_adj = partitioner.get_local_adj_tensor(c, device)
        if local_adj is None:
            continue
        model = local_models[c]
        metrics = local_evaluate(model, client_data, local_adj, device, top_k_list)
        weight = client_data["n_train_inter"] / total_inter if total_inter > 0 else 1.0 / partitioner.n_clients
        for k in top_k_list:
            all_recall[k].append(metrics[f"Recall@{k}"] * weight)
            all_ndcg[k].append(metrics[f"NDCG@{k}"] * weight)

    results = {}
    for k in top_k_list:
        results[f"Recall@{k}"] = sum(all_recall[k])
        results[f"NDCG@{k}"] = sum(all_ndcg[k])
    return results


def print_metrics(results):
    for k in sorted([int(key.split("@")[1]) for key in results if "Recall" in key]):
        rec = results.get(f"Recall@{k}", 0.0)
        ndcg = results.get(f"NDCG@{k}", 0.0)
        print(f"  Recall@{k:2d} = {rec:.6f}  NDCG@{k:2d} = {ndcg:.6f}")
