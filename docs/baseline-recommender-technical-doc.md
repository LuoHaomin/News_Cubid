# 基线推荐系统技术文档

## 目录

1. [概述](#概述)
2. [技术路线](#技术路线)
3. [基本原理](#基本原理)
4. [系统架构](#系统架构)
5. [训练流程](#训练流程)
6. [推荐生成流程](#推荐生成流程)
7. [评估方法](#评估方法)
8. [性能优化](#性能优化)
9. [模型管理](#模型管理)
10. [关键技术细节](#关键技术细节)
11. [使用指南](#使用指南)

---

## 概述

### 项目背景

本项目实现了一套基线推荐系统，用于新闻文章推荐任务。系统基于用户历史点击行为数据，预测用户未来可能点击的新闻文章，并通过MRR（平均倒数排名）指标评估推荐效果。

### 数据规模

- **训练集**：20万用户的点击日志
- **测试集**：5万用户的点击日志
- **文章库**：36万多篇新闻文章
- **Embedding维度**：250维（自动检测）

### 系统特点

- **多路召回**：结合ItemCF、UserCF和Embedding三种推荐算法
- **模型缓存**：支持模型保存和加载，避免重复训练
- **进度反馈**：实时显示训练和推荐进度
- **错误处理**：完善的异常处理和后备方案
- **性能优化**：使用稀疏矩阵、按需计算等优化策略

---

## 技术路线

### 整体架构

本系统采用**多路召回 + 结果融合**的技术路线：

```
数据加载 → 模型训练 → 多路召回 → 结果融合 → 评估输出
    ↓          ↓          ↓          ↓          ↓
训练数据    ItemCF     ItemCF推荐   投票融合    MRR计算
测试数据    UserCF     UserCF推荐   热门补充   结果输出
文章信息    Embedding  Embedding推荐 冷启动处理
Embedding
```

### 推荐策略

1. **ItemCF（物品协同过滤）**：基于物品相似度推荐
2. **UserCF（用户协同过滤）**：基于用户相似度推荐
3. **Embedding推荐**：基于文章embedding向量相似度推荐
4. **结果融合**：通过投票机制融合三种推荐结果
5. **冷启动处理**：使用热门文章处理新用户和新文章

### 技术选型

- **编程语言**：Python 3.8+
- **数据处理**：pandas, numpy
- **稀疏矩阵**：scipy.sparse
- **相似度计算**：余弦相似度
- **模型存储**：pickle + numpy/scipy格式

---

## 基本原理

### 1. ItemCF（物品协同过滤）

#### 核心思想

**"喜欢物品A的用户也喜欢物品B"** - 基于物品之间的相似度进行推荐。

#### 算法流程

1. **构建用户-物品交互矩阵**
   - 矩阵维度：`(n_users, n_articles)`
   - 矩阵元素：用户对物品的交互（点击=1，未点击=0）
   - 使用稀疏矩阵（CSR格式）存储，节省内存

2. **计算物品共现矩阵**
   ```
   Item_Cooccurrence = User_Item_Matrix^T × User_Item_Matrix
   ```
   - 矩阵维度：`(n_articles, n_articles)`
   - 矩阵元素：两个物品被同一用户点击的次数

3. **计算物品相似度矩阵**
   - 使用余弦相似度计算物品之间的相似度
   - 公式：`similarity(i, j) = (cosine_sim(i, j) + 1) / 2`
   - 将相似度归一化到[0, 1]范围

4. **生成推荐**
   - 对于用户u，获取其历史点击的物品集合H
   - 对于每个物品i ∈ H，找到与其最相似的物品集合S_i
   - 累加相似度分数：`score(j) = Σ similarity(i, j) for i in H`
   - 选择分数最高的top_k个物品作为推荐

#### 数学公式

**物品相似度（余弦相似度）**：
```
sim(i, j) = (cos(θ) + 1) / 2
其中 cos(θ) = (I_i · I_j) / (||I_i|| × ||I_j||)
I_i 是物品i的向量表示（所有用户对物品i的交互）
```

**推荐分数**：
```
score(u, j) = Σ sim(i, j) for i in clicked_items(u)
```

#### 优势与局限

**优势**：
- 物品相似度相对稳定，不需要频繁更新
- 适合物品数量相对固定的场景
- 推荐结果可解释性强

**局限**：
- 对于新物品（冷启动）效果较差
- 需要足够的用户-物品交互数据
- 计算复杂度较高（O(n_articles²)）

---

### 2. UserCF（用户协同过滤）

#### 核心思想

**"与用户A相似的用户也喜欢物品B"** - 基于用户之间的相似度进行推荐。

#### 算法流程

1. **构建用户-物品交互矩阵**
   - 矩阵维度：`(n_users, n_articles)`
   - 矩阵元素：用户对物品的交互（点击=1，未点击=0）
   - 使用稀疏矩阵存储

2. **按需计算用户相似度**（内存优化）
   - 不预先计算完整的用户相似度矩阵（节省内存）
   - 在推荐时按需计算目标用户的top-k相似用户
   - 只计算有共同交互的用户（进一步优化）

3. **生成推荐**
   - 找到与目标用户u最相似的k个用户集合S
   - 从相似用户点击的物品中推荐
   - 使用共现次数作为推荐分数

#### 数学公式

**用户相似度（余弦相似度）**：
```
sim(u, v) = (cos(θ) + 1) / 2
其中 cos(θ) = (U_u · U_v) / (||U_u|| × ||U_v||)
U_u 是用户u的向量表示（用户u对所有物品的交互）
```

**推荐分数**：
```
score(u, j) = Σ count(v, j) for v in similar_users(u)
其中 count(v, j) 表示相似用户v点击物品j的次数
```

#### 优势与局限

**优势**：
- 能够发现用户的潜在兴趣
- 推荐结果具有多样性
- 适合用户兴趣变化较快的场景

**局限**：
- 用户相似度矩阵计算复杂度高（O(n_users²)）
- 对于新用户（冷启动）效果较差
- 用户兴趣变化可能导致推荐不准确

**本系统优化**：
- 按需计算相似度，只计算top-k相似用户
- 使用分块处理，避免一次性加载所有数据
- 只计算有共同交互的用户，大幅减少计算量

---

### 3. Embedding推荐

#### 核心思想

**"相似的文章embedding向量表示相似的内容"** - 基于文章embedding向量的相似度进行推荐。

#### 算法流程

1. **加载文章Embedding向量**
   - 每篇文章有250维的embedding向量
   - Embedding向量已经过预训练，能够表示文章的语义信息

2. **计算文章相似度矩阵**
   - 使用余弦相似度计算所有文章之间的相似度
   - 公式：`similarity(i, j) = (cos(θ) + 1) / 2`
   - 矩阵维度：`(n_articles, n_articles)`

3. **生成用户Embedding**
   - 对于用户u，获取其历史点击的文章集合H
   - 计算用户embedding：`user_emb = mean(article_emb for article in H)`
   - 即用户embedding是其历史点击文章embedding的平均值

4. **生成推荐**
   - 计算用户embedding与所有文章embedding的相似度
   - 选择相似度最高的top_k个文章作为推荐

#### 数学公式

**文章相似度（余弦相似度）**：
```
sim(i, j) = (cos(θ) + 1) / 2
其中 cos(θ) = (E_i · E_j) / (||E_i|| × ||E_j||)
E_i 是文章i的embedding向量
```

**用户Embedding**：
```
user_emb = (1/|H|) × Σ E_i for i in H
其中 H 是用户历史点击的文章集合
```

**推荐分数**：
```
score(u, j) = sim(user_emb, E_j)
```

#### 优势与局限

**优势**：
- 能够捕捉文章的语义相似性
- 对于新文章（有embedding）也能推荐
- 推荐结果具有语义相关性

**局限**：
- 需要预训练的embedding向量
- 对于新用户（无历史点击）效果较差
- Embedding质量直接影响推荐效果

---

### 4. 结果融合策略

#### 融合方法

本系统采用**投票融合**策略：

1. **统计出现次数**：统计每个文章在三种推荐结果中出现的次数
2. **按次数排序**：按出现次数从高到低排序
3. **选择top_k**：选择出现次数最高的top_k个文章

#### 融合公式

```
final_score(article) = count_itemcf(article) + count_usercf(article) + count_embedding(article)
```

#### 冷启动处理

如果融合后的推荐不足top_k个：
1. 使用热门文章补充（基于训练集点击次数）
2. 排除用户已点击的文章
3. 如果仍不足，使用随机文章补充（实际应避免）

---

## 系统架构

### 模块结构

```
src/
├── main.py                 # 主入口
├── models/                 # 数据模型
│   ├── user.py            # 用户模型
│   ├── article.py         # 文章模型
│   ├── click_log.py       # 点击日志模型
│   └── recommendation.py  # 推荐结果模型
├── services/               # 业务逻辑层
│   ├── data_loader.py     # 数据加载服务
│   ├── recommender.py     # 推荐服务（融合）
│   ├── evaluator.py       # 评估服务
│   ├── output_writer.py   # 结果输出服务
│   └── model_manager.py   # 模型管理服务
├── algorithms/             # 推荐算法
│   ├── itemcf.py          # ItemCF算法
│   ├── usercf.py          # UserCF算法
│   └── embedding_based.py # Embedding推荐算法
└── utils/                  # 工具函数
    ├── config.py          # 配置管理
    ├── logger.py          # 日志管理
    ├── similarity.py      # 相似度计算
    ├── metrics.py        # 评估指标
    ├── validation.py     # 数据验证
    └── progress.py       # 进度显示
```

### 数据流

```
输入数据
  ↓
[DataLoader] 数据加载与预处理
  ↓
[Model Training] 模型训练
  ├── ItemCF.train()      → 物品相似度矩阵
  ├── UserCF.train()      → 用户-物品交互矩阵
  └── Embedding.compute() → 文章相似度矩阵
  ↓
[ModelManager] 模型保存（可选）
  ↓
[Recommender] 推荐生成
  ├── ItemCF.recommend()      → ItemCF推荐结果
  ├── UserCF.recommend()      → UserCF推荐结果
  ├── Embedding.recommend()   → Embedding推荐结果
  └── merge_recommendations() → 融合结果
  ↓
[Evaluator] 评估
  └── calculate_mrr() → MRR指标
  ↓
[OutputWriter] 结果输出
  └── write_submission() → 提交文件
```

---

## 训练流程

### 完整训练流程

#### 步骤1：数据加载

```python
# 1. 加载训练集点击日志
train_data = loader.load_train_data('data/train_click_log.csv')
# 包含字段：user_id, click_article_id, click_timestamp, ...

# 2. 加载测试集点击日志
test_data = loader.load_test_data('data/testA_click_log.csv')

# 3. 加载文章信息
articles = loader.load_articles('data/articles.csv')
# 包含字段：article_id, category_id, created_at_ts, words_count

# 4. 加载文章Embedding
embeddings = loader.load_article_embeddings('data/articles_emb.csv')
# 形状：(n_articles, 250)
```

**数据预处理**：
- 时间戳转换：将Unix时间戳转换为datetime对象
- 缺失值处理：使用默认值填充（如'unknown'、0等）
- 数据验证：检查必需字段是否存在

#### 步骤2：模型训练

##### 2.1 ItemCF模型训练

```python
itemcf = ItemCF()
item_similarity = itemcf.train(train_data, articles)
```

**详细流程**：

1. **获取文章列表**
   ```python
   article_ids = articles['article_id'].unique().tolist()
   n_articles = len(article_ids)  # 约36万
   ```

2. **构建用户-物品交互矩阵**
   ```python
   # 创建稀疏矩阵
   user_item_matrix = csr_matrix((data, (rows, cols)), 
                                  shape=(n_users, n_articles))
   # 形状：(200000, 360000)
   # 非零元素：约数百万个点击记录
   ```

3. **计算物品共现矩阵**
   ```python
   item_cooccurrence = user_item_matrix.T.dot(user_item_matrix)
   # 形状：(360000, 360000)
   # 矩阵元素：两个物品被同一用户点击的次数
   ```

4. **计算物品相似度矩阵**
   ```python
   item_similarity = cosine_similarity_sparse(item_cooccurrence)
   # 形状：(360000, 360000)
   # 使用余弦相似度，归一化到[0, 1]
   ```

**时间复杂度**：O(n_articles²)
**空间复杂度**：O(n_articles²)（稀疏矩阵）

##### 2.2 UserCF模型训练

```python
usercf = UserCF()
user_item_matrix = usercf.train(train_data)
```

**详细流程**：

1. **获取用户和文章列表**
   ```python
   user_ids = train_data['user_id'].unique().tolist()
   article_ids = train_data['click_article_id'].unique().tolist()
   n_users = len(user_ids)  # 约20万
   n_articles = len(article_ids)  # 约36万
   ```

2. **构建用户-物品交互矩阵**
   ```python
   user_item_matrix = csr_matrix((data, (rows, cols)), 
                                 shape=(n_users, n_articles))
   # 形状：(200000, 360000)
   ```

3. **不预先计算用户相似度矩阵**（内存优化）
   - 相似度将在推荐时按需计算
   - 只计算top-k相似用户
   - 只计算有共同交互的用户

**时间复杂度**：O(n_users × n_articles)（构建矩阵）
**空间复杂度**：O(n_users × n_articles)（稀疏矩阵）

##### 2.3 Embedding模型训练

```python
embedding_rec = EmbeddingRecommender()
article_ids = articles['article_id'].tolist()
embedding_rec.compute_similarity(embeddings, article_ids)
```

**详细流程**：

1. **加载Embedding向量**
   ```python
   embeddings = np.load('data/articles_emb.csv')
   # 形状：(360000, 250)
   # 使用float32以节省内存
   ```

2. **保存Embedding向量**（内存优化）
   - 不预先计算完整的文章相似度矩阵（节省内存）
   - 只保存embedding向量
   - 相似度将在推荐时按需计算

**时间复杂度**：O(n_articles × dim)（加载embedding）
**空间复杂度**：O(n_articles × dim)（只存储embedding向量）

**注意**：完整相似度矩阵需要约987 GiB内存（364047 × 364047 × 8 bytes），远超一般机器容量。因此采用按需计算策略，在推荐时直接计算用户embedding与所有文章embedding的相似度，只需O(n_articles)空间。

#### 步骤3：模型保存（可选）

```python
model_manager = ModelManager()
model_manager.save_all_models(itemcf, usercf, embedding_rec, popular_articles)
```

**保存内容**：
- ItemCF：物品相似度矩阵（.npz格式）+ 元数据（.pkl格式）
- UserCF：用户-物品交互矩阵（.npz格式）+ 元数据（.pkl格式）
- Embedding：文章相似度矩阵（.npy格式）+ 元数据（.pkl格式）
- 热门文章列表（.pkl格式）

**文件结构**：
```
data/models/
├── itemcf_model.pkl
├── itemcf_model_similarity.npz
├── usercf_model.pkl
├── usercf_model_matrix.npz
├── embedding_model.pkl
├── embedding_model_similarity.npy
└── popular_articles.pkl
```

---

## 推荐生成流程

### 完整推荐流程

#### 步骤1：获取用户历史

```python
user_history = train_data[train_data['user_id'] == user_id]['click_article_id'].tolist()
```

#### 步骤2：多路召回

##### 2.1 ItemCF推荐

```python
itemcf_recs = itemcf.recommend(user_id, user_history, top_k=5)
```

**算法流程**：

1. **获取用户历史点击的文章索引**
   ```python
   clicked_indices = [article_id_to_index[aid] for aid in user_history]
   ```

2. **计算推荐分数**
   ```python
   scores = np.zeros(n_articles)
   for clicked_idx in clicked_indices:
       similarities = item_similarity[clicked_idx, :].toarray().flatten()
       scores += similarities
   ```

3. **排除已点击文章**
   ```python
   for clicked_idx in clicked_indices:
       scores[clicked_idx] = 0.0
   ```

4. **选择top_k**
   ```python
   top_indices = np.argsort(scores)[::-1][:top_k]
   recommended_articles = [article_ids[idx] for idx in top_indices]
   ```

##### 2.2 UserCF推荐

```python
usercf_recs = usercf.recommend(user_id, None, None, train_data, top_k=5)
```

**算法流程**：

1. **按需计算相似用户**
   ```python
   top_similar_users = usercf._compute_user_similarity_on_demand(user_idx, top_k=50)
   ```

2. **从相似用户点击的文章中推荐**
   ```python
   article_scores = {}
   for similar_user_id in similar_users:
       similar_user_articles = get_user_articles(similar_user_id)
       for article_id in similar_user_articles:
           if article_id not in user_clicked:
               article_scores[article_id] += 1
   ```

3. **选择top_k**
   ```python
   sorted_articles = sorted(article_scores.items(), key=lambda x: x[1], reverse=True)
   recommended_articles = [aid for aid, score in sorted_articles[:top_k]]
   ```

##### 2.3 Embedding推荐

```python
embedding_recs = embedding_rec.recommend(user_id, user_history, top_k=5)
```

**算法流程**：

1. **计算用户Embedding**
   ```python
   user_embeddings = [embeddings[article_id_to_index[aid]] for aid in user_history]
   user_embedding = np.mean(user_embeddings, axis=0)
   user_embedding = user_embedding / np.linalg.norm(user_embedding)  # 归一化
   ```

2. **计算与所有文章的相似度**
   ```python
   scores = np.dot(normalized_embeddings, user_embedding)
   scores = (scores + 1) / 2  # 归一化到[0, 1]
   ```

3. **排除已点击文章并选择top_k**
   ```python
   for clicked_idx in clicked_indices:
       scores[clicked_idx] = 0.0
   top_indices = np.argsort(scores)[::-1][:top_k]
   recommended_articles = [article_ids[idx] for idx in top_indices]
   ```

#### 步骤3：结果融合

```python
final_recs = recommender.merge_recommendations(itemcf_recs, usercf_recs, embedding_recs, top_k=5)
```

**融合算法**：

1. **统计出现次数**
   ```python
   article_counts = {}
   for article_id in itemcf_recs:
       article_counts[article_id] = article_counts.get(article_id, 0) + 1
   for article_id in usercf_recs:
       article_counts[article_id] = article_counts.get(article_id, 0) + 1
   for article_id in embedding_recs:
       article_counts[article_id] = article_counts.get(article_id, 0) + 1
   ```

2. **按次数排序**
   ```python
   sorted_articles = sorted(article_counts.items(), key=lambda x: x[1], reverse=True)
   ```

3. **选择top_k**
   ```python
   merged_recs = [aid for aid, count in sorted_articles[:top_k]]
   ```

#### 步骤4：冷启动处理

如果融合后的推荐不足top_k个：

```python
if len(merged_recs) < top_k:
    # 使用热门文章补充
    needed = top_k - len(merged_recs)
    remaining = [aid for aid in popular_articles 
                  if aid not in user_history and aid not in merged_recs]
    merged_recs.extend(remaining[:needed])
```

---

## 评估方法

### MRR（Mean Reciprocal Rank）

#### 定义

MRR是衡量推荐系统性能的重要指标，特别适用于"只有一个正确答案"的场景。

#### 计算公式

```
MRR = (1/|U|) × Σ (1/rank_i) for i in U
```

其中：
- `U`：所有用户的集合
- `rank_i`：用户i的真实点击文章在推荐列表中的排名（如果不在推荐列表中，则rank_i = ∞，1/rank_i = 0）

#### 计算示例

假设有3个用户：

- **用户1**：真实点击文章在推荐列表的第2位 → `1/rank = 1/2 = 0.5`
- **用户2**：真实点击文章在推荐列表的第1位 → `1/rank = 1/1 = 1.0`
- **用户3**：真实点击文章不在推荐列表中 → `1/rank = 0`

```
MRR = (0.5 + 1.0 + 0) / 3 = 0.5
```


#### 指标特点

- **范围**：[0, 1]，值越大越好
- **特点**：只关注第一个正确答案的位置，适合"只有一个正确答案"的场景
- **优势**：计算简单，易于理解
- **局限**：不考虑推荐列表中的其他文章

---

## 性能优化

### 1. 内存优化

#### 稀疏矩阵

- **问题**：用户-物品交互矩阵和相似度矩阵非常稀疏（大部分元素为0）
- **解决方案**：使用`scipy.sparse.csr_matrix`存储稀疏矩阵
- **效果**：内存使用从O(n²)降低到O(nnz)，其中nnz是非零元素数量

**示例**：
```python
# 密集矩阵：360000 × 360000 × 8 bytes ≈ 1TB
# 稀疏矩阵：只存储非零元素，大幅减少内存使用
user_item_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_articles))
```

#### 按需计算

- **问题**：用户相似度矩阵太大（200000 × 200000），无法完全存储
- **解决方案**：不预先计算完整相似度矩阵，在推荐时按需计算
- **效果**：内存使用从O(n_users²)降低到O(n_users × n_articles)

**实现**：
```python
# 不预先计算
self.user_similarity = None

# 推荐时按需计算
def _compute_user_similarity_on_demand(self, user_idx, top_k=50):
    # 只计算top-k相似用户
    # 只计算有共同交互的用户
    ...
```

#### 分块处理

- **问题**：大文件一次性加载可能导致内存溢出
- **解决方案**：分块读取和处理数据
- **效果**：内存使用可控，可以处理任意大小的文件

**实现**：
```python
# 分块读取embedding文件
chunk_size = 10000
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    process_chunk(chunk)
```

### 2. 计算优化

#### 向量化计算

- **问题**：循环计算相似度效率低
- **解决方案**：使用numpy向量化操作
- **效果**：计算速度提升数十倍

**示例**：
```python
# 慢：循环计算
for i in range(n):
    for j in range(n):
        similarity[i, j] = compute_similarity(i, j)

# 快：向量化计算
similarity = cosine_similarity_dense(embeddings)
```

#### 只计算必要部分

- **问题**：计算所有用户相似度计算量大
- **解决方案**：只计算有共同交互的用户
- **效果**：计算量从O(n_users²)降低到O(n_users × avg_items_per_user)

**实现**：
```python
# 只计算有共同交互的用户
candidate_users = find_users_with_common_items(user_idx)
similarities = compute_similarity(user_idx, candidate_users)
```

### 3. 数据预处理优化

#### 共享User-Item矩阵

- **问题**：ItemCF和UserCF都构建相同的user-item交互矩阵，重复计算
- **解决方案**：在主流程中先构建一次共享矩阵，然后传递给两个算法
- **效果**：避免重复计算，节省约50%的训练时间

**实现**：
```python
# 在主流程中构建共享矩阵（只计算一次）
shared_user_item_matrix, shared_user_id_to_index, shared_article_id_to_index = \
    build_user_item_matrix(train_data, article_ids=articles['article_id'].unique().tolist())

# ItemCF使用共享矩阵
itemcf.train(train_data, articles,
              user_item_matrix=shared_user_item_matrix,
              user_id_to_index=shared_user_id_to_index,
              article_id_to_index=shared_article_id_to_index)

# UserCF使用共享矩阵
usercf.train(train_data,
              user_item_matrix=shared_user_item_matrix,
              user_id_to_index=shared_user_id_to_index,
              article_id_to_index=shared_article_id_to_index)
```

#### 预先计算用户历史

- **问题**：重复查询用户历史点击数据
- **解决方案**：预先计算所有用户的历史点击
- **效果**：避免重复查询，提升推荐生成速度

**实现**：
```python
# 预先计算
user_histories = {}
for user_id in test_users:
    user_histories[user_id] = get_user_history(user_id)

# 推荐时直接使用
for user_id in test_users:
    user_history = user_histories[user_id]
    recommendations = generate_recommendations(user_id, user_history)
```

### 4. 模型缓存

- **问题**：每次运行都需要重新训练模型，耗时数小时
- **解决方案**：保存训练好的模型，下次运行时直接加载
- **效果**：第二次运行时间从数小时降低到几分钟

**实现**：
```python
# 训练后保存
model_manager.save_all_models(itemcf, usercf, embedding_rec)

# 下次运行时加载
loaded_models = model_manager.load_all_models()
if loaded_models:
    itemcf = loaded_models['itemcf']
    usercf = loaded_models['usercf']
    embedding_rec = loaded_models['embedding']
```

---

## 模型管理

### 模型保存

#### 保存格式

- **稀疏矩阵**：使用`.npz`格式（NumPy压缩格式）
- **密集矩阵**：使用`.npy`格式（NumPy二进制格式）
- **元数据**：使用`.pkl`格式（Python pickle格式）

#### 保存内容

**ItemCF模型**：
- `itemcf_model.pkl`：文章ID列表、ID到索引的映射
- `itemcf_model_similarity.npz`：物品相似度矩阵（稀疏矩阵）

**UserCF模型**：
- `usercf_model.pkl`：用户ID列表、文章ID列表、ID到索引的映射
- `usercf_model_matrix.npz`：用户-物品交互矩阵（稀疏矩阵）

**Embedding模型**：
- `embedding_model.pkl`：文章ID列表、ID到索引的映射
- `embedding_model_embeddings.npy`：文章embedding向量矩阵（不保存相似度矩阵以节省空间）

**热门文章**：
- `popular_articles.pkl`：热门文章ID列表

### 模型加载

#### 加载流程

```python
# 1. 加载元数据
with open('itemcf_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

# 2. 加载相似度矩阵
similarity_matrix = load_npz('itemcf_model_similarity.npz')

# 3. 重建模型
itemcf = ItemCF()
itemcf.article_ids = model_data['article_ids']
itemcf.article_id_to_index = model_data['article_id_to_index']
itemcf.item_similarity = similarity_matrix
```

### 模型版本管理

**建议**：在模型文件名中添加时间戳或版本号

```python
model_name = f"itemcf_model_{timestamp}.pkl"
```

---

## 关键技术细节

### 1. 相似度计算

#### 余弦相似度

**公式**：
```
cos(θ) = (A · B) / (||A|| × ||B||)
```

**归一化到[0, 1]**：
```
similarity = (cos(θ) + 1) / 2
```

**实现**：
```python
def cosine_similarity_sparse(matrix: csr_matrix) -> csr_matrix:
    # 归一化
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1))
    norms[norms == 0] = 1
    normalized_matrix = matrix.multiply(1.0 / norms)
    
    # 计算相似度
    similarity = normalized_matrix.dot(normalized_matrix.T)
    
    # 归一化到[0, 1]并处理浮点数精度问题
    similarity.data = np.clip((similarity.data + 1) / 2, 0.0, 1.0)
    
    return similarity
```

#### 浮点数精度处理

**问题**：浮点数计算可能导致相似度值略微超出[0, 1]范围

**解决方案**：使用`np.clip`确保值在[0, 1]范围内

```python
similarity.data = np.clip((similarity.data + 1) / 2, 0.0, 1.0)
```

### 2. 冷启动处理

#### 新用户冷启动

**策略**：
1. 如果ItemCF、UserCF、Embedding都返回空结果
2. 使用热门文章推荐
3. 如果仍不足，使用随机文章补充

**实现**：
```python
if len(final_recs) < top_k:
    # 使用热门文章补充
    remaining = [aid for aid in popular_articles 
                 if aid not in user_history and aid not in final_recs]
    final_recs.extend(remaining[:needed])
```

#### 新文章冷启动

**策略**：
- ItemCF和UserCF无法处理新文章（无交互数据）
- Embedding推荐可以处理新文章（只要有embedding向量）

### 3. 数据验证

#### 输入数据验证

```python
def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]):
    """验证DataFrame是否包含必需的列"""
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {missing_columns}")
```

#### 推荐结果验证

```python
def validate_recommendation_list(article_ids: List[str], expected_length: int):
    """验证推荐结果格式"""
    if len(article_ids) != expected_length:
        raise ValueError(f"Expected {expected_length} articles, got {len(article_ids)}")
    if len(set(article_ids)) != len(article_ids):
        raise ValueError("Duplicate articles in recommendation list")
```

### 4. 可复现性

#### 随机种子设置

```python
def setup_random_seed(seed: int = 42):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
```

#### 确定性算法

- 使用固定的排序顺序
- 避免使用随机数（除非必要）
- 确保相同输入产生相同输出

---

## 使用指南

### 基本使用

```bash
# 运行推荐系统（使用默认参数）
python src/main.py
```

### 命令行参数

```bash
# 不使用缓存模型（强制重新训练）
python src/main.py --no-cache

# 不保存训练好的模型
python src/main.py --no-save

# 指定数据路径
python src/main.py \
    --train-data data/train_click_log.csv \
    --test-data data/testA_click_log.csv \
    --articles data/articles.csv \
    --embeddings data/articles_emb.csv

# 指定输出路径
python src/main.py --output data/my_submit.csv
```

### 程序化使用

```python
from src.main import run_baseline_recommender

# 运行推荐系统
mrr = run_baseline_recommender(
    train_data_path='data/train_click_log.csv',
    test_data_path='data/testA_click_log.csv',
    articles_path='data/articles.csv',
    embeddings_path='data/articles_emb.csv',
    output_path='data/submit.csv',
    use_cache=True,      # 使用模型缓存
    save_models=True    # 保存训练好的模型
)

print(f"MRR Score: {mrr:.4f}")
```

### 模型管理

```python
from src.services.model_manager import ModelManager

# 创建模型管理器
model_manager = ModelManager()

# 保存模型
model_manager.save_all_models(itemcf, usercf, embedding_rec, popular_articles)

# 加载模型
loaded_models = model_manager.load_all_models()
if loaded_models:
    itemcf = loaded_models['itemcf']
    usercf = loaded_models['usercf']
    embedding_rec = loaded_models['embedding']
```

### 性能监控

系统会自动记录：
- 数据加载时间
- 模型训练时间
- 推荐生成时间
- 评估时间
- 总处理时间

日志文件保存在`logs/baseline_recommender.log`。

---

## 总结

本基线推荐系统采用**多路召回 + 结果融合**的技术路线，结合ItemCF、UserCF和Embedding三种推荐算法，通过投票机制融合推荐结果。系统在内存优化、计算优化、模型缓存等方面做了大量优化，能够高效处理大规模数据。通过模型保存和加载机制，可以避免重复训练，大幅提升开发效率。

### 关键特性

1. **多路召回**：结合三种不同的推荐算法，提高推荐覆盖率
2. **结果融合**：通过投票机制融合多种推荐结果
3. **性能优化**：使用稀疏矩阵、按需计算等优化策略
4. **模型缓存**：支持模型保存和加载，避免重复训练
5. **进度反馈**：实时显示训练和推荐进度
6. **错误处理**：完善的异常处理和后备方案

### 适用场景

- 新闻文章推荐
- 商品推荐
- 内容推荐
- 其他基于协同过滤的推荐场景

