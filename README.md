# 新闻推荐项目

## 1. 项目概览

这个项目实现了一个端到端的新闻推荐系统，包含数据预处理、召回（ItemCF / BiNetwork / Word2Vec）、排序（pointwise + listwise）、评估和提交文件生成。
离线流程以 `--mode valid` 跑通全链路，便于观察命中率和 MRR；在线流程以 `--mode online` 运行，用于生成最终提交结果。

## 2. 目录结构

```
.
├── common/                 # 日志、并行化配置、内存压缩等通用工具
├── preprocess/             # 原始行为数据切分为 offline/online 用于训练、验证与推理
├── recall/                 # ItemCF / BiNetwork / Word2Vec 召回及汇总
├── rank/
│   ├── pointwise/          # 逐条排序特征与 LightGBM 模型训练、线上预测
│   └── listwise/           # 在 pointwise 特征基础上扩展列表特征的 LambdaMART 模型
├── eval/                   # 计算 HitRate 与 MRR 的评估函数
├── rank/utils.py           # 生成提交文件的多线程辅助工具
├── run*.sh/.ps1/.py        # 统一脚本，分别覆盖 Linux、Windows、完整流程和在线推理
├── data/                   # 原始数据（click_logs、articles）
├── user_data/
│   ├── data/               # 预处理后数据（offline/online）：click.pkl、query.pkl、recall.pkl、feature.pkl、feature_listwise.pkl
│   ├── sim/                # 保存 ItemCF / BiNetwork 相似度矩阵
│   ├── model/              # pointwise 与 listwise 模型
│   ├── log/                # 所有脚本的日志输出
│   └── tmp/                # 并行中间结果（召回、sub）
├── prediction_result/      # 线下/线上提交文件（result.csv、result_listwise.csv）
├── requirements.txt        # Python 依赖
└── 不同运行入口的 shell/PowerShell 版本
```

## 3. 数据说明

- 原始数据：`data/` 下包含 `train_click_log.csv`、`testA_click_log.csv`、`articles.csv`。
- 预处理文件保存到 `user_data/data/{offline,online}`：
  - `click.pkl`：按照时间排序的用户点击行为（纳入训练的历史行为）
  - `query.pkl`：每个用户待预测的 query（线下含正例、线上全部为 -1）
  - 召回/特征结果会逐步写入同目录（`recall_*.pkl`、`recall.pkl`、`feature.pkl`、`feature_listwise.pkl`）
- 相似度/模型/日志/提交输出分别存放在 `user_data/sim/`、`user_data/model/`、`user_data/log/`、`prediction_result/`。

## 4. 依赖与环境

1. 创建虚拟环境：`python -m venv .venv`
2. 激活虚拟环境：`source .venv/bin/activate`（Linux）或 `.\.venv\Scripts\activate`（Windows）
3. 安装依赖：`pip install -r requirements.txt`
4. 项目根目录需要加入 `PYTHONPATH`，运行脚本时由 `run.*` 自动设置。

## 5. 核心运行流程（离线验证）

**步骤顺序**
1. 数据预处理（`preprocess/data.py --mode valid`）
2. ItemCF / BiNetwork / Word2Vec 召回生成局部候选（`recall/recall_*.py --mode valid`）
3. 合并召回并打分（`recall/recall.py --mode valid`）
4. Pointwise 特征工程（`rank/pointwise/rank_feature.py --mode valid`）
5. LightGBM pointwise 排序训练（`rank/pointwise/rank_lgb.py --mode valid`）
6. Listwise 特征扩展（`rank/listwise/rank_feature_listwise.py --mode valid`）
7. LambdaMART listwise 训练并生成提交（`rank/listwise/rank_lambdamart.py --mode valid`）

**推荐用法**
- Linux：`bash run.sh`（会依次运行上述 1~7 步，并把日志写入 `user_data/log/{timestamp}.log`）
- Windows：`.\run.ps1`（同样的步骤与日志机制）
- 纯 Python：`python run_full_pipeline.py` 可生成同样的日志文件名 `full_pipeline_{timestamp}.log`。


**手动单阶段命令示例**
```bash
python preprocess/data.py --mode valid --logfile valid.log
python recall/recall_itemcf.py --mode valid --logfile valid.log
python recall/recall_binetwork.py --mode valid --logfile valid.log
python recall/recall_w2v.py --mode valid --logfile valid.log
python recall/recall.py --mode valid --logfile valid.log
python rank/pointwise/rank_feature.py --mode valid --logfile valid.log
python rank/pointwise/rank_lgb.py --mode valid --logfile valid.log
python rank/listwise/rank_feature_listwise.py --mode valid --logfile valid.log
python rank/listwise/rank_lambdamart.py --mode valid --logfile valid.log
```

## 6. 在线推理流程（`--mode online`）

- 使用 `run_online.sh`（Linux）或 `run_online.ps1`（Windows）运行完整的在线推理，自动打点日志。
- 该流程会依次运行数据预处理、召回、特征、Pointwise 预测和 Listwise 预测（需要先训练好的模型）。
- 线上结果写入 `prediction_result/result.csv`（pointwise）与 `prediction_result/result_listwise.csv`（listwise）。

## 7. 模块说明

- `common/`：日志、Pandarallel/Multitasking 的线程/进程控制、`reduce_mem_usage` 内存压缩。
- `preprocess/data.py`：将 `data/train_click_log.csv` 与 `data/testA_click_log.csv` 划分为 `offline`/`online` 的 `click.pkl` 与 `query.pkl`。
- `recall/recall_itemcf.py`、`recall_binetwork.py`、`recall_w2v.py`：分别构建召回候选池并记录 `sim_score` 与 `label`，保存到 `user_data/data/{offline,online}`；也生成用于特征工程的相似度字典。
- `recall/recall.py`：标准化 `sim_score`、加权融合召回结果、删除无正例用户、根据 `common.utils.Logger` 打印召回间相似度，并在离线模式下评估 HitRate/MRR。
- `rank/pointwise/rank_feature.py`：基于 recall、article metadata、历史行为统计、ItemCF/BiNetwork/W2V 相似度等特征，生成 `feature.pkl`。
- `rank/pointwise/rank_lgb.py`：使用 GroupKFold 的 5 折 LightGBM 分类器训练、保存模型、输出 `prediction_result/result.csv`，内部调用 `rank/utils.gen_sub` 生成 top5 提交。
- `rank/listwise/rank_feature_listwise.py`：在 pointwise 特征基础上拓展列表统计特征（类目多样性、sim_score 统计、rank 等），会自动回退调用 pointwise 脚本确保 `feature.pkl` 存在。
- `rank/listwise/rank_lambdamart.py`：训练 LambdaMART 模型、保存 `user_data/model/listwise/lambdamart{fold}.pkl`、对验证集/测试集输出 `result_listwise.csv`，可在在线模式执行预测。
- `eval/evaluate.py`：提供 HitRate@K 与 MRR@K（K=5/10/20/40/50）评估函数。
- `rank/utils.py`：多线程 `gen_sub` 用于构建 Top5 提交列表，补全缺失物品并输出 `prediction_result` 的 CSV。
- `check_progress.py`：快速检查每个阶段的输出文件是否生成，便于调试。

## 8. 结果路径

- 模型：`user_data/model/`（pointwise `lgb*.pkl`，listwise `listwise/lambdamart*.pkl`）
- 日志：`user_data/log/{timestamp}.log`
- 预测提交：`prediction_result/result.csv`（pointwise）、`prediction_result/result_listwise.csv`（listwise）
- 召回/特征缓存：`user_data/data/{offline,online}`、`user_data/sim/{offline,online}`、`user_data/tmp/`。

## 9. 检查与调试

- `python check_progress.py`：验证 `user_data` 中 click/recall/feature 文件、模型数量，输出结果文件大小。
- 若某一阶段失败，先查 `user_data/log/{timestamp}.log`，日志里会打印已完成路径与估算的平均召回长度。

## 10. 小贴士

1. 所有脚本都依赖 `PYTHONPATH` 包含项目根目录；`run.sh`、`run.ps1` 等脚本会自动完成这一步。
2. 离线流程使用 `--mode valid` 并会评估 HitRate/MRR，线上流程使用 `--mode online` 并生成可提交文件。
3. `run_full_pipeline.py` 是平台无 Shell 时的替代方案，内部依次调用各阶段的 Python 命令并打印耗时。# 新闻推荐项目

