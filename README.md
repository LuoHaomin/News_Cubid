# 数据科学-新闻推荐

[零基础入门推荐系统 - 新闻推荐_GETSTART_Competition introduction-Tianchi competition-Alibabacloud Tianchi description & data](https://tianchi.aliyun.com/competition/entrance/531842/information)

## 目录
- [数据科学-新闻推荐](#数据科学-新闻推荐)
  - [目录](#目录)
  - [项目概述](#项目概述)
  - [快速开始](#快速开始)
    - [环境要求](#环境要求)
    - [安装依赖](#安装依赖)
    - [数据准备](#数据准备)
    - [运行推荐系统](#运行推荐系统)
  - [项目结构](#项目结构)
  - [技术栈](#技术栈)
  - [数据](#数据)
    - [数据表](#数据表)
    - [字段表](#字段表)
  - [提交样例](#提交样例)
  - [评分方式](#评分方式)
  - [技术方案](#技术方案)
    - [多路召回](#多路召回)
      - [ItemCF召回（@luohaomin）](#itemcf召回luohaomin)
        - [物品特征提取](#物品特征提取)
      - [UserCF召回](#usercf召回)
      - [基于Embedding的召回](#基于embedding的召回)
    - [排序](#排序)



## 项目概述

本项目实现了一套基线推荐系统，用于新闻文章推荐任务。系统基于用户历史点击行为数据，预测用户未来可能点击的新闻文章，并通过MRR（平均倒数排名）指标评估推荐效果。

## 快速开始

### 环境要求
- Python 3.8+
- 8GB+ 内存（推荐16GB）
- 足够的磁盘空间（至少2GB用于数据文件）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据准备

确保以下数据文件位于`data/`目录下：
- `train_click_log.csv` - 训练集用户点击日志
- `testA_click_log.csv` - 测试集用户点击日志
- `articles.csv` - 文章信息数据表
- `articles_emb.csv` - 文章embedding向量表示
- `sample_submit.csv` - 提交样例文件（格式参考）

### 运行推荐系统

**基本用法**:
```bash
python src/main.py
```

默认使用**本地验证模式**：从训练数据中分割验证集（每个用户的最后一条点击作为验证数据，之前的数据作为训练数据）。这将处理所有用户，可能需要8小时左右。

**本地验证模式（默认）**:
```bash
# 默认使用本地验证：从训练数据中分割验证集
python src/main.py

# 只处理前100个用户进行快速测试（约几分钟）
python src/main.py --test-users 100
```

**使用外部测试集（用于最终提交）**:
```bash
# 使用外部测试集（testA_click_log.csv）进行最终提交
python src/main.py --external-test
```

**快速测试模式（推荐用于开发和调试）**:

如果只想快速测试和评估，可以限制处理的用户数量：

```bash
# 只处理前100个用户进行快速测试（约几分钟）
python src/main.py --test-users 100

# 只处理前1000个用户（约半小时）
python src/main.py --test-users 1000
```

**其他命令行参数**:
```bash
# 不使用缓存模型（强制重新训练）
python src/main.py --no-cache

# 不保存训练好的模型
python src/main.py --no-save

# 指定输出路径
python src.main.py --output data/my_submit.csv

# 组合使用：快速测试模式 + 不使用缓存
python src/main.py --test-users 100 --no-cache
```

**验证模式说明**:
- **本地验证模式（默认）**：从训练数据中分割验证集，每个用户的最后一条点击作为验证数据。这种方式可以：
  - 避免冷启动问题（测试用户都有训练历史）
  - 更准确地评估模型性能
  - 适合模型开发和调试
  
- **外部测试集模式**：使用外部测试集（testA_click_log.csv）进行评估。这种方式：
  - 用于最终提交
  - 测试用户可能是新用户（冷启动）
  - 更接近真实场景

**模型缓存**:
- 系统会自动保存训练好的模型到 `data/models/` 目录
- 下次运行时，如果检测到已保存的模型，会自动加载，避免重复训练
- 使用 `--no-cache` 可以强制重新训练

## 项目结构

```
src/
├── models/              # 数据模型和实体类
├── services/           # 业务逻辑层
├── algorithms/         # 推荐算法实现
└── utils/              # 工具函数

tests/
├── unit/               # 单元测试
├── integration/       # 集成测试
└── fixtures/           # 测试数据
```

## 技术栈

- Python 3.8+
- pandas, numpy - 数据处理
- scikit-learn, scipy - 推荐算法和相似度计算
- pytest - 测试框架


## 数据

赛题以预测用户未来点击新闻文章为任务，数据集报名后可见并可下载，该数据来自某新闻APP平台的用户交互数据，包括30万用户，近300万次点击，共36万多篇不同的新闻文章，同时每篇新闻文章有对应的embedding向量表示。为了保证比赛的公平性，将会从中抽取20万用户的点击日志数据作为训练集，5万用户的点击日志数据作为测试集A，5万用户的点击日志数据作为测试集B。

### 数据表

**`train_click_log.csv`**：训练集用户点击日志

**`testA_click_log.csv`**：测试集用户点击日志

**`articles.csv`**：新闻文章信息数据表

**`articles_emb.csv`**：新闻文章embedding向量表示

**`sample_submit.csv`**：提交样例文件

### 字段表

| **Field**               | **Description**                  |
| ----------------------- | -------------------------------- |
| user_id                 | 用户id                           |
| click_article_id        | 点击文章id                       |
| click_timestamp         | 点击时间戳                       |
| click_environment       | 点击环境                         |
| click_deviceGroup       | 点击设备组                       |
| click_os                | 点击操作系统                     |
| click_country           | 点击城市                         |
| click_region            | 点击地区                         |
| click_referrer_type     | 点击来源类型                     |
| article_id              | 文章id，与click_article_id相对应 |
| category_id             | 文章类型id                       |
| created_at_ts           | 文章创建时间戳                   |
| words_count             | 文章字数                         |
| emb_1,emb_2,...,emb_249 | 文章embedding向量表示            |

| 名称                | 大小                                           | Link                                                                               |
| ------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------- |
| articles.csv        | 9.89MB                                         | <http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531842/articles.csv>        |
| articles_emb.csv    | 973.15MB(MD5:1f8a7fc79e0ad13311e27e3408d0287b) | <http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531842/articles_emb.csv>    |
| testA_click_log.csv | 20.47MB                                        | <http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531842/testA_click_log.csv> |
| train_click_log.csv | 43.5MB                                         | <http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531842/train_click_log.csv> |

## 提交样例

user_id,article_1,article_2,article_3,article_4,article_5

根据用户历史浏览点击新闻的数据信息预测用户最后一次点击的新闻文章。

## 评分方式

**`MRR(Mean Reciprocal Rank)`**

## 技术方案

### 多路召回

#### ItemCF召回（@luohaomin）

##### 物品特征提取

信息源：文章的embedding向量表示、文章的信息（类别、创建时间、字数）

#### UserCF召回

#### 基于Embedding的召回

### 排序
