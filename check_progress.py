#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查流程进度"""
import os
from pathlib import Path

root_dir = Path(__file__).parent
user_data_dir = root_dir / 'user_data'

def check_file(path, name):
    exists = path.exists()
    size = path.stat().st_size / (1024 * 1024) if exists else 0
    status = "✓" if exists else "✗"
    print(f"{status} {name}: {'存在' if exists else '不存在'} ({size:.2f} MB)" if exists else f"{status} {name}: 不存在")
    return exists

print("=" * 60)
print("流程进度检查")
print("=" * 60)

# 数据预处理
print("\n【数据预处理】")
check_file(user_data_dir / 'data' / 'offline' / 'click.pkl', 'click.pkl')
check_file(user_data_dir / 'data' / 'offline' / 'query.pkl', 'query.pkl')

# 召回阶段
print("\n【召回阶段】")
check_file(user_data_dir / 'sim' / 'offline' / 'itemcf_sim.pkl', 'itemcf_sim.pkl')
check_file(user_data_dir / 'data' / 'offline' / 'recall_itemcf.pkl', 'recall_itemcf.pkl')
check_file(user_data_dir / 'sim' / 'offline' / 'binetwork_sim.pkl', 'binetwork_sim.pkl')
check_file(user_data_dir / 'data' / 'offline' / 'recall_binetwork.pkl', 'recall_binetwork.pkl')
check_file(user_data_dir / 'data' / 'offline' / 'article_w2v.pkl', 'article_w2v.pkl')
check_file(user_data_dir / 'data' / 'offline' / 'recall_w2v.pkl', 'recall_w2v.pkl')
check_file(user_data_dir / 'data' / 'offline' / 'recall.pkl', 'recall.pkl (合并)')

# 特征工程
print("\n【特征工程】")
check_file(user_data_dir / 'data' / 'offline' / 'feature.pkl', 'feature.pkl')

# 模型训练
print("\n【模型训练】")
model_dir = user_data_dir / 'model'
if model_dir.exists():
    model_files = list(model_dir.glob('lgb*.pkl'))
    print(f"✓ 模型文件: {len(model_files)} 个")
    for mf in model_files:
        size = mf.stat().st_size / (1024 * 1024)
        print(f"  - {mf.name}: {size:.2f} MB")
else:
    print("✗ 模型目录不存在")

# 预测结果
print("\n【预测结果】")
check_file(root_dir / 'prediction_result' / 'result.csv', 'result.csv')

print("\n" + "=" * 60)

