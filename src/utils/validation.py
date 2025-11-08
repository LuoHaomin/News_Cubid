"""
数据验证工具模块
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from .exceptions import DataValidationError


def validate_user_id(user_id: str) -> None:
    """
    验证用户ID
    
    Args:
        user_id: 用户ID
        
    Raises:
        DataValidationError: 如果用户ID无效
    """
    if not user_id or not isinstance(user_id, str) or not user_id.strip():
        raise DataValidationError(f"Invalid user_id: {user_id}")


def validate_article_id(article_id: str) -> None:
    """
    验证文章ID
    
    Args:
        article_id: 文章ID
        
    Raises:
        DataValidationError: 如果文章ID无效
    """
    # 如果article_id不是字符串，尝试转换为字符串
    if not isinstance(article_id, str):
        if article_id is None:
            raise DataValidationError(f"Invalid article_id: {article_id}")
        try:
            article_id = str(article_id)
        except Exception:
            raise DataValidationError(f"Invalid article_id: {article_id}")
    
    if not article_id or not article_id.strip():
        raise DataValidationError(f"Invalid article_id: {article_id}")


def validate_timestamp(timestamp: int) -> None:
    """
    验证时间戳
    
    Args:
        timestamp: 时间戳
        
    Raises:
        DataValidationError: 如果时间戳无效
    """
    if not isinstance(timestamp, (int, np.integer)) or timestamp <= 0:
        raise DataValidationError(f"Invalid timestamp: {timestamp}")


def validate_embedding(embedding: np.ndarray, expected_dim: int = None) -> None:
    """
    验证embedding向量
    
    Args:
        embedding: embedding向量
        expected_dim: 期望的维度（可选，如果为None则不检查维度）
        
    Raises:
        DataValidationError: 如果embedding向量无效
    """
    if not isinstance(embedding, np.ndarray):
        raise DataValidationError(f"Embedding must be numpy array, got {type(embedding)}")
    if expected_dim is not None and embedding.shape != (expected_dim,):
        raise DataValidationError(f"Embedding dimension mismatch: expected {expected_dim}, got {embedding.shape}")
    if len(embedding.shape) != 1 or embedding.shape[0] <= 0:
        raise DataValidationError(f"Invalid embedding shape: {embedding.shape}")


def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    """
    验证DataFrame是否包含必需的列
    
    Args:
        df: 要验证的DataFrame
        required_columns: 必需的列名列表
        
    Raises:
        DataValidationError: 如果缺少必需的列
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {missing_columns}")


def validate_recommendation_list(article_ids: List[str], top_k: int = 5) -> None:
    """
    验证推荐列表
    
    Args:
        article_ids: 推荐文章ID列表
        top_k: 期望的文章数量，默认5
        
    Raises:
        DataValidationError: 如果推荐列表无效
    """
    if not isinstance(article_ids, list):
        raise DataValidationError(f"Recommendation list must be a list, got {type(article_ids)}")
    if len(article_ids) != top_k:
        raise DataValidationError(f"Recommendation list must contain exactly {top_k} articles, got {len(article_ids)}")
    if len(set(article_ids)) != len(article_ids):
        raise DataValidationError("Recommendation list contains duplicate article IDs")
    for article_id in article_ids:
        validate_article_id(article_id)

