#!/bin/bash

time=$(date "+%Y-%m-%d-%H:%M:%S")

# 环境准备
echo "环境准备"
# source ./env/bin/activate
export PYTHONPATH=.:$PYTHONPATH

rm -rf user_data/model/
rm -rf user_data/data/offline/
rm -rf user_data/data/online/
rm -rf user_data/sim/
rm -rf user_data/tmp/

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

# 排序特征（Pointwise 基础特征）
echo "排序特征"
python rank/pointwise/rank_feature.py --mode valid --logfile "${time}.log"

# Listwise 特征工程
echo "Listwise 特征"
python rank/listwise/rank_feature_listwise.py --mode valid --logfile "${time}.log"

# LambdaMART 排序模型训练
echo "LambdaMART 训练"
python rank/listwise/rank_lambdamart.py --mode valid --logfile "${time}.log"

# 在线预测
# echo "在线预测"
# python rank/listwise/rank_lambdamart.py --mode online --logfile "${time}.log"