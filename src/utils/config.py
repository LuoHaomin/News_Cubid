"""
配置模块：提供随机种子和路径配置
"""

import os
import random
import numpy as np

# 随机种子配置（用于可复现性）
RANDOM_SEED = 42

# 数据路径配置
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
MODEL_DIR = os.path.join(DATA_DIR, 'models')
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_click_log.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'testA_click_log.csv')
ARTICLES_PATH = os.path.join(DATA_DIR, 'articles.csv')
ARTICLES_EMB_PATH = os.path.join(DATA_DIR, 'articles_emb.csv')
SAMPLE_SUBMIT_PATH = os.path.join(DATA_DIR, 'sample_submit.csv')

# 创建模型目录（如果不存在）
os.makedirs(MODEL_DIR, exist_ok=True)

# 推荐系统配置
TOP_K = 5  # 每个用户推荐的文章数量
EMBEDDING_DIM = None  # embedding向量维度（自动检测，实际数据可能是250维）

# 测试模式配置（用于快速测试和评估）
# 设置为None或0表示使用所有测试用户，设置为正整数表示只处理前N个用户
# 例如：TEST_USER_LIMIT = 100  # 只处理前100个用户进行快速测试
TEST_USER_LIMIT = None  # 默认使用所有用户，可通过环境变量或命令行参数覆盖


def setup_random_seed(seed: int = RANDOM_SEED) -> None:
    """
    设置随机种子以确保可复现性
    
    Args:
        seed: 随机种子值，默认42
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# 初始化时设置随机种子
setup_random_seed()

