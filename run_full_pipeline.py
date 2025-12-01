#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运行完整的推荐系统流程"""
import os
import subprocess
import sys
import time
from datetime import datetime

# 设置环境变量
os.environ['PYTHONPATH'] = '.'

def run_command(cmd, description):
    """运行命令并显示进度"""
    print(f"\n{'='*60}")
    print(f"开始执行: {description}")
    print(f"命令: {cmd}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        elapsed = time.time() - start_time
        print(f"\n✓ {description} 完成 (耗时: {elapsed:.2f}秒)")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {description} 失败 (耗时: {elapsed:.2f}秒)")
        print(f"错误代码: {e.returncode}")
        return False

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    logfile = f"full_pipeline_{timestamp}.log"
    
    print(f"开始运行完整流程，日志文件: {logfile}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    steps = [
        # 数据预处理
        ("python preprocess/data.py --mode valid --logfile " + logfile, "数据预处理"),
        
        # 召回阶段
        ("python recall/recall_itemcf.py --mode valid --logfile " + logfile, "ItemCF召回"),
        ("python recall/recall_binetwork.py --mode valid --logfile " + logfile, "BiNetwork召回"),
        ("python recall/recall_w2v.py --mode valid --logfile " + logfile, "Word2Vec召回"),
        ("python recall/recall.py --mode valid --logfile " + logfile, "召回合并"),
        
        # 特征工程
        ("python rank/pointwise/rank_feature.py --mode valid --logfile " + logfile, "特征工程"),
        
        # 模型训练
        ("python rank/pointwise/rank_lgb.py --mode valid --logfile " + logfile, "LightGBM模型训练"),
    ]
    
    success_count = 0
    for i, (cmd, desc) in enumerate(steps, 1):
        print(f"\n步骤 {i}/{len(steps)}: {desc}")
        if run_command(cmd, desc):
            success_count += 1
        else:
            print(f"\n流程在步骤 {i} ({desc}) 失败，停止执行")
            break
    
    print(f"\n{'='*60}")
    print(f"流程完成: {success_count}/{len(steps)} 步骤成功")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    if success_count == len(steps):
        print("✓ 所有步骤执行成功！")
        return 0
    else:
        print("✗ 部分步骤失败，请检查日志")
        return 1

if __name__ == '__main__':
    sys.exit(main())

