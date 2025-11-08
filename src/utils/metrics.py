"""
评估指标计算模块
"""

from typing import Dict, List
import numpy as np
from .exceptions import EvaluationError


def calculate_mrr(recommendations: Dict[str, List[str]], 
                  ground_truth: Dict[str, str]) -> float:
    """
    计算MRR（平均倒数排名）指标
    
    Args:
        recommendations: 推荐结果字典，key为用户ID，value为推荐文章ID列表
        ground_truth: 真实标签字典，key为用户ID，value为真实点击文章ID
        
    Returns:
        MRR值，范围[0, 1]
        
    Raises:
        EvaluationError: 如果输入数据无效
    """
    if not recommendations or not ground_truth:
        raise EvaluationError("Recommendations and ground_truth cannot be empty")
    
    if set(recommendations.keys()) != set(ground_truth.keys()):
        raise EvaluationError("Recommendations and ground_truth must have the same user IDs")
    
    reciprocal_ranks = []
    
    for user_id, rec_list in recommendations.items():
        true_article = ground_truth.get(user_id)
        if not true_article:
            reciprocal_ranks.append(0.0)
            continue
        
        # 找到真实文章在推荐列表中的位置（从1开始）
        try:
            rank = rec_list.index(true_article) + 1
            reciprocal_rank = 1.0 / rank
        except ValueError:
            # 真实文章不在推荐列表中
            reciprocal_rank = 0.0
        
        reciprocal_ranks.append(reciprocal_rank)
    
    # 计算平均倒数排名
    mrr = np.mean(reciprocal_ranks)
    
    # 验证MRR值在[0, 1]范围内
    if mrr < 0.0 or mrr > 1.0:
        raise EvaluationError(f"MRR value out of range [0, 1]: {mrr}")
    
    return float(mrr)


def calculate_hit_rate(recommendations: Dict[str, List[str]], 
                       ground_truth: Dict[str, str]) -> float:
    """
    计算Hit Rate指标（命中率）
    
    Args:
        recommendations: 推荐结果字典
        ground_truth: 真实标签字典
        
    Returns:
        Hit Rate值，范围[0, 1]
    """
    if not recommendations or not ground_truth:
        return 0.0
    
    hits = 0
    total = 0
    
    for user_id, rec_list in recommendations.items():
        true_article = ground_truth.get(user_id)
        if true_article:
            total += 1
            if true_article in rec_list:
                hits += 1
    
    return hits / total if total > 0 else 0.0


def calculate_precision_at_k(recommendations: Dict[str, List[str]], 
                             ground_truth: Dict[str, str], 
                             k: int = 5) -> float:
    """
    计算Precision@K指标
    
    Args:
        recommendations: 推荐结果字典
        ground_truth: 真实标签字典
        k: 考虑前k个推荐结果
        
    Returns:
        Precision@K值，范围[0, 1]
    """
    if not recommendations or not ground_truth:
        return 0.0
    
    precisions = []
    
    for user_id, rec_list in recommendations.items():
        true_article = ground_truth.get(user_id)
        if not true_article:
            continue
        
        # 取前k个推荐结果
        top_k_recs = rec_list[:k]
        
        # 计算命中数
        hits = 1 if true_article in top_k_recs else 0
        precision = hits / len(top_k_recs) if top_k_recs else 0.0
        
        precisions.append(precision)
    
    return np.mean(precisions) if precisions else 0.0

