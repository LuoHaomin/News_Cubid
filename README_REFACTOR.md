# 天池新闻推荐项目重构说明

## 项目结构

项目已重构为模块化结构：

```
DataScience/
├── common/              # 公共工具模块
│   ├── __init__.py
│   └── utils.py         # Logger 类和 reduce_mem_usage 函数
├── preprocess/          # 数据预处理模块
│   ├── __init__.py
│   └── data.py          # 数据准备和划分
├── recall/              # 召回模块
│   ├── __init__.py
│   ├── recall_itemcf.py      # ItemCF 召回
│   ├── recall_binetwork.py   # BiNetwork 召回
│   ├── recall_w2v.py         # Word2Vec 召回
│   └── recall.py              # 召回合并
├── sorting/             # 排序模块
│   ├── __init__.py
│   ├── utils.py         # gen_sub 相关函数
│   ├── rank_feature.py   # 特征工程
│   └── rank_lgb.py       # LightGBM 排序模型
├── eval/                 # 评估模块
│   ├── __init__.py
│   └── evaluate.py      # 评估函数
└── run.sh               # 运行脚本
```

## 运行方式

在项目根目录执行：

```bash
bash run.sh
```

或者分步执行：

```bash
# 1. 数据预处理
python preprocess/data.py --mode valid --logfile "test.log"

# 2. 召回
python recall/recall_itemcf.py --mode valid --logfile "test.log"
python recall/recall_binetwork.py --mode valid --logfile "test.log"
python recall/recall_w2v.py --mode valid --logfile "test.log"
python recall/recall.py --mode valid --logfile "test.log"

# 3. 特征工程
python sorting/rank_feature.py --mode valid --logfile "test.log"

# 4. 模型训练
python sorting/rank_lgb.py --mode valid --logfile "test.log"
```

## 数据路径

- 原始数据：`data/` 目录
  - `train_click_log.csv`
  - `testA_click_log.csv`
  - `articles.csv`

- 中间数据：`user_data/` 目录
  - `user_data/data/offline/` - 离线验证数据
  - `user_data/data/online/` - 在线测试数据
  - `user_data/sim/` - 相似度矩阵
  - `user_data/model/` - 模型文件
  - `user_data/tmp/` - 临时文件
  - `user_data/log/` - 日志文件

- 预测结果：`prediction_result/` 目录

## 模块说明

### common
公共工具模块，包含：
- `Logger`: 日志记录类
- `reduce_mem_usage`: 内存优化函数

### preprocess
数据预处理模块，负责：
- 数据加载和清洗
- 训练/验证集划分
- 数据保存

### recall
召回模块，包含三种召回方法：
- ItemCF: 基于物品协同过滤
- BiNetwork: 基于双向网络
- Word2Vec: 基于词向量

### sorting
排序模块，包含：
- 特征工程：构建排序特征
- LightGBM 模型：训练和预测

### eval
评估模块，包含：
- `evaluate`: 计算 HitRate 和 MRR 指标

## 注意事项

1. 所有脚本使用相对路径 `../data/` 和 `../user_data/`，确保从项目根目录运行
2. 日志文件保存在 `user_data/log/` 目录
3. 模型文件保存在 `user_data/model/` 目录
4. 预测结果保存在 `prediction_result/` 目录

