"""
用户-物品交互矩阵构建工具
用于在ItemCF和UserCF之间共享同一个矩阵，避免重复计算
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple, Dict, List

from .logger import logger


def build_user_item_matrix(
    click_log: pd.DataFrame,
    article_ids: List[str] = None,
    user_ids: List[str] = None
) -> Tuple[csr_matrix, Dict[str, int], Dict[str, int]]:
    """
    构建用户-物品交互矩阵（共享版本）
    
    Args:
        click_log: 用户点击日志
        article_ids: 文章ID列表（可选，如果为None则从click_log中提取）
        user_ids: 用户ID列表（可选，如果为None则从click_log中提取）
        
    Returns:
        (user_item_matrix, user_id_to_index, article_id_to_index)
        - user_item_matrix: 用户-物品交互矩阵（稀疏矩阵）
        - user_id_to_index: 用户ID到索引的映射
        - article_id_to_index: 文章ID到索引的映射
    """
    logger.info("Building shared user-item interaction matrix...")
    
    # 获取用户ID列表
    if user_ids is None:
        user_ids = click_log['user_id'].unique().tolist()
    user_id_to_index = {uid: idx for idx, uid in enumerate(user_ids)}
    n_users = len(user_ids)
    logger.info(f"Found {n_users} users")
    
    # 获取文章ID列表
    if article_ids is None:
        article_ids = click_log['click_article_id'].unique().tolist()
    article_id_to_index = {aid: idx for idx, aid in enumerate(article_ids)}
    n_articles = len(article_ids)
    logger.info(f"Found {n_articles} articles")
    
    # 构建稀疏矩阵
    rows = []
    cols = []
    data = []
    
    total_rows = len(click_log)
    for idx, (_, row) in enumerate(click_log.iterrows()):
        if (idx + 1) % 100000 == 0:
            logger.info(f"  Processing click logs: {idx + 1}/{total_rows} ({100*(idx+1)/total_rows:.1f}%)")
        
        user_id = row['user_id']
        article_id = row['click_article_id']
        
        # 只处理在索引中的用户和文章
        if user_id in user_id_to_index and article_id in article_id_to_index:
            user_idx = user_id_to_index[user_id]
            article_idx = article_id_to_index[article_id]
            rows.append(user_idx)
            cols.append(article_idx)
            data.append(1.0)  # 二进制交互
    
    logger.info("Creating sparse matrix...")
    user_item_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_articles), dtype=np.float32)
    logger.info(f"User-item matrix created: shape {user_item_matrix.shape}, "
               f"non-zero elements: {user_item_matrix.nnz}")
    
    return user_item_matrix, user_id_to_index, article_id_to_index

