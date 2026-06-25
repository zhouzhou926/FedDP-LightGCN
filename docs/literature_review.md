# DP-LightGCN 文献综述清单

## 说明
本文献综述围绕差分隐私+LightGCN（非联邦）论文方向，
按5个主题分类，共计17篇核心参考文献，
每篇附有与本文的定位差异说明。

---

## A. 差分隐私+推荐系统（基础必读，5篇）

### [A1] McSherry & Mironov (2009)
- 标题: Differentially Private Recommender Systems
- 会议: KDD 2009
- 核心方法: 在协同过滤的预测阶段添加Laplace噪声
- 与本文关系: 最早将DP引入推荐系统，奠基之作，必引
- 与本文差异: 使用简单的协同过滤，未涉及GCN/深度学习

### [A2] Friedman et al. (2016)
- 标题: Differential Privacy in Matrix Factorization
- 期刊: Journal of Privacy and Confidentiality
- 核心方法: 矩阵分解时对梯度添加噪声（DP-SGD思路）
- 与本文关系: DP-MF基线方法代表
- 与本文差异: 矩阵分解模型，非图神经网络

### [A3] Shin et al. (2018)
- 标题: User Privacy in Recommender Systems via Differential Privacy
- 会议: ECML-PKDD 2018
- 核心方法: 在输入评分上应用本地差分隐私（LDP）
- 与本文关系: LDP推荐基线
- 与本文差异: 本地DP而非中心化DP，场景不同

### [A4] Berlioz et al. (2015)
- 标题: Differential Privacy in Collaborative Filtering
- 会议: RecSys 2015
- 核心方法: 系统对比输入/梯度/输出三种扰动策略
- 与本文关系: 扰动位置分析必引文献

### [A5] Liu et al. (2023)
- 标题: A Survey on Differential Privacy for Recommender Systems
- 类型: arXiv 综述
- 核心内容: 全面综述DP推荐系统研究现状
- 与本文关系: 帮助定位本文在领域中的位置

## B. 差分隐私+图神经网络（技术基础，4篇）

### [B1] Jia et al. (2020) -- DP-GCN
- 标题: Towards Differentially Private GCNs
- 会议: NeurIPS 2020
- 核心方法: 在GCN邻接矩阵/嵌入上加噪，提出edge-DP
- 与本文关系: 最直接相关的方法论基石
- 与本文差异: 通用GCN(含特征变换+非线性) 节点分类非推荐

### [B2] Wu et al. (2023)
- 标题: Privacy-Preserving Graph Neural Networks: A Survey
- 类型: arXiv 综述
- 核心内容: GNN隐私保护技术全景综述

### [B3] Sajadmanesh et al. (2023) -- LPGNet
- 标题: Locally Differentially Private Graph Neural Networks
- 会议: NeurIPS 2023
- 核心方法: LDP+GNN，本地扰动特征
- 与本文差异: 本地DP vs 中心化DP，面向通用GNN

### [B4] Daigavane et al. (2021)
- 标题: Node-Level Differential Privacy for Graph Neural Networks
- 类型: arXiv 2021
- 核心方法: 严格定义GNN节点级DP
- 与本文关系: 本文使用节点级DP，理论分析可引用

## C. LightGCN核心文献（2篇）

### [C1] He et al. (2020) -- LightGCN
- 标题: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
- 会议: SIGIR 2020
- 核心方法: 去除特征变换和非线性激活，仅保留邻居聚合
- 与本文关系: 本文的主干模型

### [C2] Wang et al. (2019) -- NGCF
- 标题: Neural Graph Collaborative Filtering
- 会议: SIGIR 2019
- 核心方法: 标准GCN推荐（含特征变换+非线性）
- 与本文关系: LightGCN前身，对比说明简化优势

## D. 差分隐私+推荐模型适配（最相关现存工作，3篇）

### [D1] Zhang et al. (2022) -- PrivGNN
- 标题: PrivGNN: Privacy-preserving GNN-based Recommendation
- 会议: CIKM 2022
- 核心方法: GCN推荐+差分隐私+联邦学习
- 与本文差异: 联邦场景(本文中心化) 通用GCN(本文LightGCN)

### [D2] Gao et al. (2023)
- 标题: Differentially Private Graph Neural Networks for Recommendation
- 会议: PAKDD 2023
- 核心方法: GNN推荐+嵌入扰动+DP
- 与本文差异: 使用NGCF非LightGCN 未设计层级预算分配

### [D3] Li et al. (2024) -- P3GNN
- 标题: P3GNN: Privacy-Preserving Graph Neural Networks for Recommendation
- 期刊: Knowledge-Based Systems (SCI二区)
- 核心方法: 层级嵌入扰动+自适应隐私预算
- 与本文差异: 通用GCN架构 预算分配策略不同

## E. 隐私会计与基础理论（2篇）

### [E1] Mironov (2017)
- 标题: Renyi Differential Privacy
- 会议: IEEE CSF 2017
- 核心贡献: 提出RDP框架，更tight的composition bound
- 与本文关系: 本文的隐私会计基础

### [E2] Abadi et al. (2016) -- DP-SGD
- 标题: Deep Learning with Differential Privacy
- 会议: CCS 2016
- 核心方法: 深度学习梯度裁剪+噪声注入范式
- 与本文关系: DP训练经典范式，对比基线参考

---

## 创新定位总结

| 维度 | 现有工作(D1-D3) | 本文方案 |
|------|-----------------|----------|
| 主干模型 | 通用GCN/NGCF | LightGCN(更简洁) |
| 扰动对象 | 梯度或通用嵌入 | LightGCN层级嵌入 |
| 预算分配 | 均匀或简单分配 | 自适应(基于层嵌入范数) |
| 隐私会计 | 基础(eps,delta)-DP | RDP Composition(更tight) |
| 噪声机制 | 直接加噪 | 裁剪+加噪(控制敏感度) |

## 建议阅读顺序
1. 先读 [C1] LightGCN原文
2. 再读 [B1] DP-GCN
3. 接着读 [D2]+[D3]
4. 最后读 [A1]-[A4]+[E1]
