# 系统数据流分析文档

## 概述

本文档详细分析基线推荐系统中所有数据的计算时机、存储方式、大小估算和依赖关系。

---

## 数据分类

### 1. 输入数据（从文件加载）

| 数据名称 | 来源文件 | 大小估算 | 格式 | 说明 |
|---------|---------|---------|------|------|
| **训练集点击日志** | `train_click_log.csv` | ~43.5 MB | CSV | 20万用户，111万条点击记录 |
| **测试集点击日志** | `testA_click_log.csv` | ~20.5 MB | CSV | 5万用户，51.8万条点击记录 |
| **文章信息** | `articles.csv` | ~9.9 MB | CSV | 36.4万篇文章的基本信息 |
| **文章Embedding** | `articles_emb.csv` | ~973 MB | CSV | 36.4万篇文章，每篇250维embedding |

**总输入数据大小**：约 **1.05 GB**

---

## 训练阶段数据

### 2. 训练时计算的数据（内存中，不保存）

#### 2.1 共享User-Item交互矩阵

| 数据名称 | 计算时机 | 大小估算 | 格式 | 生命周期 |
|---------|---------|---------|------|---------|
| **共享User-Item矩阵** | 训练开始时 | ~4.4 MB | `csr_matrix` (float32) | 训练期间，训练后释放 |
| **用户ID到索引映射** | 训练开始时 | ~1.6 MB | `dict` | 训练期间，训练后释放 |
| **文章ID到索引映射** | 训练开始时 | ~2.9 MB | `dict` | 训练期间，训练后释放 |

**计算方式**：
```python
# 只计算一次，ItemCF和UserCF共享
shared_user_item_matrix, shared_user_id_to_index, shared_article_id_to_index = \
    build_user_item_matrix(train_data, article_ids=articles['article_id'].unique().tolist())
```

**大小计算**：
- 矩阵形状：`(200000, 364047)`
- 非零元素：约110万
- 稀疏矩阵存储：`110万 × (4 bytes索引 + 4 bytes索引 + 4 bytes值) ≈ 13.2 MB`
- 实际CSR格式：约 **4.4 MB**（压缩后）

**依赖关系**：
- 依赖：`train_data`（训练集点击日志）
- 被依赖：ItemCF训练、UserCF训练

---

#### 2.2 ItemCF训练数据

| 数据名称 | 计算时机 | 大小估算 | 格式 | 生命周期 |
|---------|---------|---------|------|---------|
| **物品共现矩阵** | ItemCF训练时 | ~63 MB | `csr_matrix` | **保存到磁盘** |
| **用户-物品交互矩阵** | ItemCF训练时 | ~4.4 MB | `csr_matrix` | **保存到磁盘** |
| **物品相似度矩阵** | 不计算 | - | - | 不保存（内存优化） |

**计算流程**：
```python
# 1. 使用共享矩阵计算物品共现矩阵（保存）
item_cooccurrence = user_item_matrix.T.dot(user_item_matrix)
# 形状：(364047, 364047)，非零元素：约416万
# 大小：416万 × 12 bytes ≈ 50 MB（实际压缩后约63 MB）

# 2. 不预先计算完整的物品相似度矩阵（内存优化）
# 相似度将在推荐时按需计算
self.item_cooccurrence = item_cooccurrence
self.user_item_matrix = user_item_matrix
self.item_similarity = None
```

**大小计算**：
- 物品共现矩阵：`416万非零元素 × 12 bytes ≈ 50 MB`（实际约63 MB）
- 用户-物品交互矩阵：`110万非零元素 × 12 bytes ≈ 13.2 MB`（实际压缩后约4.4 MB）
- 如果计算完整相似度矩阵：`2.94亿非零元素 × 12 bytes ≈ 3.5 GB`（**不计算**）

**依赖关系**：
- 依赖：共享User-Item矩阵
- 被依赖：ItemCF推荐（按需计算相似度）

---

#### 2.3 UserCF训练数据

| 数据名称 | 计算时机 | 大小估算 | 格式 | 生命周期 |
|---------|---------|---------|------|---------|
| **用户-物品交互矩阵** | UserCF训练时 | ~4.4 MB | `csr_matrix` (float32) | **保存到磁盘** |
| **用户相似度矩阵** | 推荐时按需计算 | 不保存 | 按需计算 | 每次推荐时临时计算 |

**计算流程**：
```python
# 直接使用共享矩阵（不重新计算）
self.user_item_matrix = shared_user_item_matrix
# 形状：(200000, 364047)，非零元素：约110万
# 大小：约4.4 MB（保存）
```

**依赖关系**：
- 依赖：共享User-Item矩阵
- 被依赖：UserCF推荐（按需计算用户相似度）

---

#### 2.4 Embedding训练数据

| 数据名称 | 计算时机 | 大小估算 | 格式 | 生命周期 |
|---------|---------|---------|------|---------|
| **文章Embedding向量** | 训练时加载 | ~364 MB | `numpy.ndarray` (float32) | **保存到磁盘** |
| **文章相似度矩阵** | 不计算 | - | - | 不保存（内存优化） |

**计算流程**：
```python
# 只保存embedding向量，不计算相似度矩阵
self.article_embeddings = embeddings.astype(np.float32)
# 形状：(364047, 250)
# 大小：364047 × 250 × 4 bytes ≈ 364 MB（保存）
```

**大小计算**：
- 文章Embedding：`364047 × 250 × 4 bytes ≈ 364 MB`
- 如果计算完整相似度矩阵：`364047 × 364047 × 8 bytes ≈ 987 GB`（**不计算**）

**依赖关系**：
- 依赖：`articles_emb.csv`（输入文件）
- 被依赖：Embedding推荐（推荐时按需计算相似度）

---

#### 2.5 热门文章列表

| 数据名称 | 计算时机 | 大小估算 | 格式 | 生命周期 |
|---------|---------|---------|------|---------|
| **热门文章列表** | 训练时计算 | ~4 KB | `list` | **保存到磁盘** |

**计算流程**：
```python
article_counts = train_data['click_article_id'].value_counts()
popular_articles = article_counts.head(100).index.tolist()
# 大小：100个文章ID × 约40 bytes ≈ 4 KB
```

**依赖关系**：
- 依赖：`train_data`（训练集点击日志）
- 被依赖：冷启动处理

---

### 3. 保存到磁盘的数据（模型文件）

| 数据名称 | 保存文件 | 大小估算 | 格式 | 说明 |
|---------|---------|---------|------|------|
| **ItemCF共现矩阵** | `itemcf_model_cooccurrence.npz` | ~63 MB | 稀疏矩阵 | 物品共现矩阵（用于按需计算相似度） |
| **ItemCF交互矩阵** | `itemcf_model_matrix.npz` | ~4.4 MB | 稀疏矩阵 | 用户-物品交互矩阵（用于按需计算相似度） |
| **ItemCF元数据** | `itemcf_model.pkl` | ~3 MB | Pickle | 文章ID列表、索引映射 |
| **UserCF交互矩阵** | `usercf_model_matrix.npz` | ~4.4 MB | 稀疏矩阵 | 用户-物品交互矩阵 |
| **UserCF元数据** | `usercf_model.pkl` | ~4.5 MB | Pickle | 用户ID列表、文章ID列表、索引映射 |
| **Embedding向量** | `embedding_model_embeddings.npy` | ~364 MB | NumPy数组 | 文章embedding向量 |
| **Embedding元数据** | `embedding_model.pkl` | ~3 MB | Pickle | 文章ID列表、索引映射 |
| **热门文章列表** | `popular_articles.pkl` | ~4 KB | Pickle | 热门文章ID列表 |

**总保存数据大小**：约 **448 MB**（相比之前的2.6 GB，节省约83%）

**文件结构**：
```
data/models/
├── itemcf_model.pkl                    (~3 MB)
├── itemcf_model_cooccurrence.npz       (~63 MB) ⭐ 最大文件
├── itemcf_model_matrix.npz            (~4.4 MB)
├── usercf_model.pkl                    (~4.5 MB)
├── usercf_model_matrix.npz            (~4.4 MB)
├── embedding_model.pkl                 (~3 MB)
├── embedding_model_embeddings.npy      (~364 MB)
└── popular_articles.pkl                (~4 KB)
```

---

## 推荐阶段数据

### 4. 推荐时现场计算的数据（不保存）

#### 4.1 ItemCF推荐

| 数据名称 | 计算时机 | 大小估算 | 格式 | 说明 |
|---------|---------|---------|------|------|
| **物品相似度** | 每次推荐时按需计算 | ~几KB到几MB | `numpy.ndarray` | 只计算有共现的物品的相似度 |
| **用户推荐分数** | 每次推荐时 | ~1.5 MB | `numpy.ndarray` | 每个用户36.4万篇文章的分数 |
| **Top-K文章索引** | 每次推荐时 | ~40 bytes | `numpy.ndarray` | Top-5文章索引 |

**计算流程**：
```python
# 1. 按需计算用户历史点击文章的相似文章（只计算有共现的物品）
scores = np.zeros(n_articles)  # 36.4万 × 4 bytes ≈ 1.5 MB
for clicked_idx in clicked_indices:
    # 从共现矩阵中提取共现信息，按需计算相似度
    similarities = _compute_item_similarity_on_demand(clicked_idx)
    scores += similarities

# 2. 选择top-k
top_indices = np.argsort(scores)[::-1][:top_k]  # 5 × 8 bytes ≈ 40 bytes
```

**依赖关系**：
- 依赖：ItemCF共现矩阵（已保存）、用户-物品交互矩阵（已保存）、用户历史点击
- 输出：推荐文章列表

**优化效果**：
- 不保存相似度矩阵：2.2 GB → 0 GB（节省100%存储）
- 只保存共现矩阵：63 MB（用于按需计算相似度）
- 推荐时按需计算相似度，只计算有共现的物品，大幅减少计算量

---

#### 4.2 UserCF推荐

| 数据名称 | 计算时机 | 大小估算 | 格式 | 说明 |
|---------|---------|---------|------|------|
| **相似用户索引** | 每次推荐时 | ~200 bytes | `numpy.ndarray` | Top-50相似用户索引 |
| **用户相似度分数** | 每次推荐时 | ~200 bytes | `numpy.ndarray` | 50个相似用户的相似度 |
| **文章推荐分数** | 每次推荐时 | ~几KB | `dict` | 文章ID到分数的映射 |

**计算流程**：
```python
# 1. 按需计算top-k相似用户（只计算有共同交互的用户）
top_similar_users = _compute_user_similarity_on_demand(user_idx, top_k=50)
# 大小：50 × 4 bytes ≈ 200 bytes

# 2. 从相似用户点击的文章中推荐
article_scores = {}  # 动态大小，通常几KB
for similar_user_id in similar_users:
    similar_user_articles = get_user_articles(similar_user_id)
    for article_id in similar_user_articles:
        article_scores[article_id] += 1
```

**依赖关系**：
- 依赖：UserCF交互矩阵（已保存）、训练集点击日志
- 输出：推荐文章列表

---

#### 4.3 Embedding推荐

| 数据名称 | 计算时机 | 大小估算 | 格式 | 说明 |
|---------|---------|---------|------|------|
| **用户Embedding** | 每次推荐时 | ~1 KB | `numpy.ndarray` | 250维向量 |
| **归一化文章Embedding** | 每次推荐时 | ~364 MB | `numpy.ndarray` | 所有文章的归一化embedding |
| **推荐分数** | 每次推荐时 | ~1.5 MB | `numpy.ndarray` | 36.4万篇文章的分数 |

**计算流程**：
```python
# 1. 计算用户embedding（平均历史点击文章的embedding）
user_embedding = np.mean(user_embeddings, axis=0)  # 250 × 4 bytes ≈ 1 KB

# 2. 归一化所有文章embedding（可优化：预先归一化）
normalized_embeddings = article_embeddings / article_norms  # 364 MB（临时）

# 3. 计算相似度分数
scores = np.dot(normalized_embeddings, user_embedding)  # 36.4万 × 4 bytes ≈ 1.5 MB
```

**依赖关系**：
- 依赖：文章Embedding向量（已保存）、用户历史点击
- 输出：推荐文章列表

**优化建议**：可以预先归一化文章embedding，避免每次推荐时重复计算。

---

#### 4.4 结果融合

| 数据名称 | 计算时机 | 大小估算 | 格式 | 说明 |
|---------|---------|---------|------|------|
| **文章出现次数** | 每次推荐时 | ~几KB | `dict` | 文章ID到出现次数的映射 |
| **融合后推荐列表** | 每次推荐时 | ~200 bytes | `list` | Top-5文章ID列表 |

**计算流程**：
```python
# 统计每个文章在三种推荐结果中出现的次数
article_counts = {}  # 动态大小，通常几KB
for article_id in itemcf_recs + usercf_recs + embedding_recs:
    article_counts[article_id] = article_counts.get(article_id, 0) + 1

# 按次数排序，选择top-k
merged_recs = sorted(article_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
```

**依赖关系**：
- 依赖：ItemCF推荐结果、UserCF推荐结果、Embedding推荐结果
- 输出：最终推荐文章列表

---

### 5. 评估阶段数据

| 数据名称 | 计算时机 | 大小估算 | 格式 | 说明 |
|---------|---------|---------|------|------|
| **Ground Truth** | 评估时计算 | ~2 MB | `dict` | 用户ID到真实点击文章的映射 |
| **MRR分数** | 评估时计算 | ~8 bytes | `float` | 单个MRR值 |

**计算流程**：
```python
# 提取每个用户的最后一次点击作为ground truth
ground_truth = {}  # 5万用户 × 40 bytes ≈ 2 MB
for user_id, group in test_click_log.groupby('user_id'):
    last_click = group.sort_values('click_timestamp').iloc[-1]
    ground_truth[user_id] = last_click['click_article_id']

# 计算MRR
mrr = calculate_mrr(recommendations, ground_truth)  # 单个float值
```

**依赖关系**：
- 依赖：推荐结果、测试集点击日志
- 输出：MRR评估指标

---

## 数据依赖关系图

```
输入数据（1.05 GB）
├── train_click_log.csv (43.5 MB)
│   ├── → 共享User-Item矩阵 (4.4 MB) [临时]
│   │   ├── → ItemCF训练
│   │   │   ├── → 物品共现矩阵 (63 MB) [临时]
│   │   │   └── → 物品相似度矩阵 (2.2 GB) [保存] ⭐
│   │   └── → UserCF训练
│   │       └── → 用户-物品交互矩阵 (4.4 MB) [保存]
│   └── → 热门文章列表 (4 KB) [保存]
│
├── articles.csv (9.9 MB)
│   └── → 文章ID列表
│       ├── → ItemCF训练（文章索引）
│       └── → Embedding训练（文章索引）
│
└── articles_emb.csv (973 MB)
    └── → 文章Embedding向量 (364 MB) [保存]

推荐阶段（现场计算）
├── ItemCF推荐
│   ├── 依赖：物品相似度矩阵 (2.2 GB) [已保存]
│   ├── 依赖：用户历史点击 [临时]
│   └── 计算：用户推荐分数 (1.5 MB) [临时]
│
├── UserCF推荐
│   ├── 依赖：用户-物品交互矩阵 (4.4 MB) [已保存]
│   ├── 依赖：训练集点击日志 [临时]
│   └── 计算：相似用户、文章分数 [临时]
│
├── Embedding推荐
│   ├── 依赖：文章Embedding向量 (364 MB) [已保存]
│   ├── 依赖：用户历史点击 [临时]
│   └── 计算：用户embedding、相似度分数 (1.5 MB) [临时]
│
└── 结果融合
    ├── 依赖：三种推荐结果 [临时]
    └── 计算：融合后推荐列表 [临时]
```

---

## 数据大小汇总

### 输入数据（从文件加载）
- 训练集点击日志：43.5 MB
- 测试集点击日志：20.5 MB
- 文章信息：9.9 MB
- 文章Embedding：973 MB
- **总计**：**1.05 GB**

### 保存的模型数据（磁盘）
- ItemCF共现矩阵：63 MB
- ItemCF交互矩阵：4.4 MB
- ItemCF元数据：3 MB
- UserCF交互矩阵：4.4 MB
- UserCF元数据：4.5 MB
- Embedding向量：364 MB
- Embedding元数据：3 MB
- 热门文章列表：4 KB
- **总计**：**约 448 MB**（相比之前的2.6 GB，节省约83%）

### 训练时临时数据（内存，不保存）
- 共享User-Item矩阵：4.4 MB
- 用户ID映射：1.6 MB
- 文章ID映射：2.9 MB
- 物品共现矩阵：63 MB
- **总计**：**约 72 MB**

### 推荐时临时数据（内存，每次推荐）
- ItemCF推荐分数：1.5 MB
- UserCF相似用户：~200 bytes
- Embedding归一化：364 MB（可优化）
- Embedding推荐分数：1.5 MB
- 融合结果：~几KB
- **总计**：**约 367 MB**（每次推荐）

---

## 关键发现

### 1. 最大数据文件
- **Embedding向量**：364 MB（占保存数据的81%）
- **ItemCF共现矩阵**：63 MB（占保存数据的14%）
- 所有模型文件都在1GB以下

### 2. 内存优化策略
- **Embedding相似度矩阵**：不保存（987 GB → 0 GB）
  - 改为推荐时按需计算用户embedding与所有文章的相似度
  - 节省约987 GB存储空间

- **UserCF用户相似度矩阵**：不保存（约800 GB → 0 GB）
  - 改为推荐时按需计算top-k相似用户
  - 节省约800 GB存储空间

- **ItemCF物品相似度矩阵**：不保存（2.2 GB → 0 GB）
  - 改为推荐时按需计算物品相似度（只计算有共现的物品）
  - 只保存物品共现矩阵（63 MB）和用户-物品交互矩阵（4.4 MB）
  - 节省约2.2 GB存储空间

### 3. 可优化点

#### 3.1 Embedding归一化优化
**问题**：每次推荐时都归一化所有文章的embedding（364 MB）

**优化方案**：
```python
# 训练时预先归一化
normalized_embeddings = article_embeddings / article_norms
# 保存归一化后的embedding
```

**效果**：推荐时直接使用，避免重复计算，节省约364 MB临时内存

#### 3.2 用户历史预计算
**问题**：每次推荐时都查询用户历史点击

**优化方案**：
```python
# 推荐前预先计算所有用户的历史
user_histories = {}
for user_id in test_users:
    user_histories[user_id] = get_user_history(user_id)
```

**效果**：避免重复查询，提升推荐速度

---

## 数据生命周期

### 训练阶段
1. **加载输入数据**（1.05 GB）→ 内存
2. **构建共享矩阵**（4.4 MB）→ 内存，训练后释放
3. **训练ItemCF** → 生成相似度矩阵（2.2 GB）→ 保存到磁盘
4. **训练UserCF** → 生成交互矩阵（4.4 MB）→ 保存到磁盘
5. **训练Embedding** → 保存embedding向量（364 MB）→ 保存到磁盘
6. **计算热门文章**（4 KB）→ 保存到磁盘

### 推荐阶段（使用保存的模型）
1. **加载模型**（2.6 GB）→ 内存
2. **为每个用户推荐**：
   - ItemCF：使用相似度矩阵计算分数（1.5 MB临时）
   - UserCF：按需计算相似用户（~200 bytes临时）
   - Embedding：按需计算相似度（1.5 MB临时）
   - 融合结果（~几KB临时）
3. **释放临时数据** → 内存

### 评估阶段
1. **提取Ground Truth**（2 MB）→ 内存
2. **计算MRR**（8 bytes）→ 内存
3. **输出结果** → 磁盘

---

## 存储空间需求

### 最小存储需求（仅输入数据）
- **输入数据**：1.05 GB
- **模型文件**：0 GB（不保存模型）
- **总计**：**1.05 GB**

### 推荐存储需求（保存模型）
- **输入数据**：1.05 GB
- **模型文件**：448 MB
- **总计**：**约 1.5 GB**（相比之前的3.65 GB，节省约59%）

### 运行内存需求
- **训练阶段**：约 **3.5 GB**（输入数据 + 临时数据 + 模型数据）
- **推荐阶段**：约 **3.0 GB**（模型数据 + 临时数据）

---

## 总结

### 保存的数据（448 MB）
1. **Embedding向量**（364 MB）- 最大文件
2. **ItemCF共现矩阵**（63 MB）
3. **ItemCF交互矩阵**（4.4 MB）
4. **UserCF交互矩阵**（4.4 MB）
5. **元数据**（约10.5 MB）
6. **热门文章列表**（4 KB）

### 现场计算的数据
1. **ItemCF物品相似度** - 每次推荐时按需计算（只计算有共现的物品）
2. **UserCF用户相似度** - 每次推荐时按需计算
3. **Embedding相似度** - 每次推荐时按需计算
4. **推荐分数** - 每次推荐时计算
5. **结果融合** - 每次推荐时计算

### 优化效果
- **Embedding相似度矩阵**：987 GB → 0 GB（节省99.9%存储）
- **UserCF相似度矩阵**：800 GB → 0 GB（节省100%存储）
- **ItemCF相似度矩阵**：2.2 GB → 0 GB（节省100%存储）
- **总存储空间**：2.6 GB → 448 MB（节省约83%存储）
- **共享User-Item矩阵**：避免重复计算（节省50%训练时间）

---

**文档版本**：v1.0  
**最后更新**：2024-12-19

