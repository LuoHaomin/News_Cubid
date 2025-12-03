# 排序层

排序层负责对召回阶段得到的候选新闻进行精排，根据用户特征和新闻特征预测用户对每条新闻的点击概率，并按照概率从高到低排序。

## pointwise（已实现）

Pointwise 方法将排序问题转化为二分类问题，对每个用户-新闻对独立预测点击概率。

### 文件结构

- `rank_feature.py`: 特征工程脚本
- `rank_lgb.py`: LightGBM 模型训练和预测脚本

### 特征工程 (`rank_feature.py`)

#### 1. 文章基础特征

- `article_id`: 文章ID
- `category_id`: 文章类别ID
- `words_count`: 文章字数
- `created_at_ts`: 文章创建时间戳
- `created_at_datetime`: 文章创建时间（datetime格式）

#### 2. 用户历史点击统计特征

- **时间相关特征**:
  - `user_id_click_article_created_at_ts_diff_mean`: 用户点击文章的创建时间差的平均值
  - `user_id_click_diff_mean`: 用户点击时间差的平均值
  - `user_click_timestamp_created_at_ts_diff_mean/std`: 点击时间与文章创建时间差的均值/标准差
  - `user_click_datetime_hour_std`: 用户点击时间小时数的标准差
  - `user_click_last_article_created_time`: 用户最后点击的文章创建时间
  - `user_click_last_article_click_time`: 用户最后点击时间
  - `user_last_click_created_at_ts_diff`: 当前文章创建时间与用户最后点击文章创建时间的差值
  - `user_last_click_timestamp_diff`: 当前文章创建时间与用户最后点击时间的差值

- **内容相关特征**:
  - `user_clicked_article_words_count_mean`: 用户点击文章字数的平均值
  - `user_click_last_article_words_count`: 用户最后点击文章的字数
  - `user_last_click_words_count_diff`: 当前文章字数与用户最后点击文章字数的差值

- **计数统计特征**:
  - `user_id_cnt`: 用户点击次数
  - `article_id_cnt`: 文章被点击次数
  - `user_id_category_id_cnt`: 用户对某类别的点击次数

#### 3. 召回相关特征

- **ItemCF 相似度特征**:
  - `user_clicked_article_itemcf_sim_sum`: 用户历史点击文章与当前文章的ItemCF相似度加权和（时间衰减因子0.7）
  - `user_last_click_article_itemcf_sim`: 用户最后点击文章与当前文章的ItemCF相似度

- **BiNetwork 相似度特征**:
  - `user_last_click_article_binetwork_sim`: 用户最后点击文章与当前文章的BiNetwork相似度

- **Word2Vec 相似度特征**:
  - `user_last_click_article_w2v_sim`: 用户最后点击文章与当前文章的Word2Vec余弦相似度
  - `user_click_article_w2w_sim_sum_2`: 用户历史点击文章与当前文章的Word2Vec相似度加权和（取最近2条）

#### 4. 召回分数特征

- `sim_score`: 召回阶段的综合相似度分数（来自 `recall.pkl`）

### 模型训练 (`rank_lgb.py`)

#### 模型配置

- **算法**: LightGBM (LGBMClassifier)
- **参数**:
  - `num_leaves`: 64
  - `max_depth`: 10
  - `learning_rate`: 0.05
  - `n_estimators`: 10000
  - `subsample`: 0.8
  - `feature_fraction`: 0.8
  - `reg_alpha`: 0.5
  - `reg_lambda`: 0.5
  - `random_state`: 2020

#### 训练流程

1. **数据划分**: 使用 `GroupKFold(n_splits=5)` 进行5折交叉验证，按 `user_id` 分组确保同一用户不会同时出现在训练集和验证集
2. **断点续训**:
   - 检查模型文件是否存在，如果存在则跳过训练直接加载
   - 保存验证集索引到 `fold_indices.pkl`，避免重复划分
3. **模型训练**:
   - 使用早停机制（`early_stopping_rounds=100`）
   - 每100轮输出一次训练日志
   - 保存训练好的模型到 `user_data/model/lgb{fold_id}.pkl`
4. **预测与评估**:
   - 对验证集进行预测，计算 OOF (Out-of-Fold) 预测
   - 对测试集进行预测，5折模型结果平均
   - 计算评估指标：HitRate@5/10/20/40/50 和 MRR@5/10/20/40/50
5. **生成提交文件**: 使用 `gen_sub()` 函数生成最终提交结果

#### 评估指标

- **HitRate@K**: 前K个推荐中命中真实点击的比例
- **MRR@K** (Mean Reciprocal Rank): 真实点击在前K个推荐中的平均倒数排名

### 使用方法

```bash
# 1. 特征工程
python rank/pointwise/rank_feature.py --mode valid --logfile "test_feature.log"

# 2. 模型训练
python rank/pointwise/rank_lgb.py --mode valid --logfile "test_lgb.log"

# 3. 在线预测（使用训练好的模型）
python rank/pointwise/rank_lgb.py --mode online --logfile "test_online.log"
```

### 输出文件

- `user_data/data/{mode}/feature.pkl`: 特征文件
- `user_data/model/lgb{0-4}.pkl`: 5折模型文件
- `user_data/model/fold_indices.pkl`: 验证集索引文件（用于断点续训）
- `prediction_result/result.csv`: 最终提交文件

## pairwise（未实现）

Pairwise 方法将排序问题转化为样本对的相对顺序问题，学习判断两个新闻中哪个更可能被用户点击。

### 设计思路（pairwise）

1. **数据构造**:
   - 对于每个用户，构造正负样本对（点击的新闻 vs 未点击的新闻）
   - 标签：1 表示第一个新闻更相关，0 表示第二个新闻更相关

2. **模型选择**:
   - 可以使用 RankNet、LambdaRank 等 pairwise 排序算法
   - 或使用 LightGBM 的 `lambdarank` 目标函数

3. **特征复用**:
   - 可以复用 pointwise 中的大部分特征
   - 额外构造相对特征：两个新闻之间的特征差值

4. **训练方式**:
   - 使用 pairwise loss（如 pairwise logistic loss）
   - 考虑使用 LambdaRank 优化 NDCG 等排序指标

### 预期文件结构（pairwise）

```text
pairwise/
├── rank_feature_pairwise.py  # 构造样本对和相对特征
├── rank_lambdarank.py        # LambdaRank 模型训练
└── rank_ranknet.py           # RankNet 模型训练（可选）
```

## listwise（未实现）

Listwise 方法直接优化整个列表的排序质量，考虑列表中所有新闻的相对关系。

### 设计思路（listwise）

1. **数据构造**:
   - 对每个用户，构造一个包含多个候选新闻的列表
   - 标签：每个新闻的相关性分数或排序位置

2. **模型选择**:
   - ListNet: 使用排列概率分布作为学习目标
   - ListMLE: 使用最大似然估计优化列表排序
   - LambdaMART: 结合 LambdaRank 和 MART（梯度提升树）

3. **特征设计**:
   - 复用 pointwise 特征
   - 添加列表级别的特征：列表中新闻的多样性、覆盖率等

4. **训练目标**:
   - 直接优化 NDCG、MAP 等列表级别的排序指标
   - 考虑使用 ListNet 的排列概率损失或 ListMLE 的似然损失

### 预期文件结构（listwise）

```text
listwise/
├── rank_feature_listwise.py  # 构造列表特征
├── rank_listnet.py          # ListNet 模型训练
├── rank_listmle.py          # ListMLE 模型训练（可选）
└── rank_lambdamart.py       # LambdaMART 模型训练
```

## 工具函数 (`utils.py`)

- `gen_sub(prediction)`: 生成提交文件
  - 对每个用户，选择预测概率最高的5个新闻
  - 如果不足5个，随机补充其他新闻
  - 支持多线程并行处理

## 参考资源

- **Pointwise**: 将排序问题转化为分类/回归问题
- **Pairwise**: RankNet, LambdaRank
- **Listwise**: ListNet, ListMLE, LambdaMART
