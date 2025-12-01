#!/bin/bash

time=$(date "+%Y-%m-%d-%H:%M:%S")

# 处理数据
echo "处理数据"
python preprocess/data.py --mode valid --logfile "${time}.log"

# itemcf 召回
echo "itemcf 召回"
python recall/recall_itemcf.py --mode valid --logfile "${time}.log"

# binetwork 召回
echo "binetwork 召回"
python recall/recall_binetwork.py --mode valid --logfile "${time}.log"

# w2v 召回
echo "w2v 召回"
python recall/recall_w2v.py --mode valid --logfile "${time}.log"

# 召回合并
echo "召回合并"
python recall/recall.py --mode valid --logfile "${time}.log"

# 排序特征
echo "排序特征"
python rank/pointwise/rank_feature.py --mode valid --logfile "${time}.log"

# lgb 模型训练
echo "lgb 模型训练"
python rank/pointwise/rank_lgb.py --mode valid --logfile "${time}.log"

