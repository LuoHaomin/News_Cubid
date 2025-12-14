## 赛题简介

赛题以预测用户未来点击新闻文章为任务，数据集报名后可见并可下载，该数据来自某新闻APP平台的用户交互数据，包括 30 万用户，近 300 万次点击，共 36 万多篇不同的新闻文章，同时每篇新闻文章有对应的 embedding 向量表示。为了保证比赛的公平性，将会从中抽取 20 万用户的点击日志数据作为训练集，5 万用户的点击日志数据作为测试集 A，5 万用户的点击日志数据作为测试集 B。

---

## 数据表

- **train_click_log.csv**：训练集用户点击日志
- **testA_click_log.csv**：测试集用户点击日志
- **articles.csv**：新闻文章信息数据表
- **articles_emb.csv**：新闻文章 embedding 向量表示
- **sample_submit.csv**：提交样例文件

---

## 字段表

| Field                | Description             |
|----------------------|------------------------|
| user_id              | 用户 id                |
| click_article_id     | 点击文章 id            |
| click_timestamp      | 点击时间戳             |
| click_environment    | 点击环境               |
| click_deviceGroup    | 点击设备组             |
| click_os             | 点击操作系统           |
| click_country        | 点击城市               |
| click_region         | 点击地区               |
| click_referrer_type  | 点击来源类型           |
| article_id           | 文章 id，与 click_article_id 相对应 |
| category_id          | 文章类型 id            |
| created_at_ts        | 文章创建时间戳         |
| words_count          | 文章字数               |
| emb_1,emb_2,...,emb_249 | 文章 embedding 向量表示 |

---

## 结果提交

提交前请确保预测结果的格式与 `sample_submit.csv` 中的格式一致，以及提交文件后缀名为 csv。其格式如下：

```
user_id,article_1,article_2,article_3,article_4,article_5
```

其中 `user_id` 为用户 id，`article_1,article_2,article_3,article_4,article_5` 为预测用户点击新闻文章 Top5 的 `article_id`，依概率从高到低排序，例如：

```
user_id,article_1,article_2,article_3,article_4,article_5
200000,1,2,3,4,5
200001,1,2,3,4,5
200002,1,2,3,4,5
200003,1,2,3,4,5
```

---

## 评分方式

**MRR (Mean Reciprocal Rank)**：首先对选手提交的表格中的每个用户计算用户得分。

公式如下：

$$
score(user)=\sum_{k=1}^{5} \frac{s(user, k)}{k}
$$

其中，如果选手对该 user 的预测结果 predict k 命中该 user 的最后一条购买数据则 $s(user,k)=1$；否则 $s(user,k)=0$。而选手得分为所有这些 score(user) 的平均值。