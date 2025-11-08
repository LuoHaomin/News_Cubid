"""
文章模型
"""

import numpy as np
from typing import Optional


class Article:
    """文章实体类"""
    
    def __init__(self, 
                 article_id: str,
                 category_id: int = 0,
                 created_at_ts: int = 0,
                 words_count: int = 0,
                 embedding: Optional[np.ndarray] = None):
        """
        初始化文章
        
        Args:
            article_id: 文章唯一标识符
            category_id: 文章类别ID
            created_at_ts: 文章创建时间戳
            words_count: 文章字数
                 embedding: 文章embedding向量（维度自动检测）
        """
        self.article_id = article_id
        self.category_id = category_id
        self.created_at_ts = created_at_ts
        self.words_count = words_count
        
        # 如果embedding缺失，使用零向量填充（维度从实际数据中获取）
        if embedding is None:
            # 默认使用250维（根据实际数据）
            self.embedding = np.zeros(250, dtype=np.float32)
        else:
            self.embedding = embedding
    
    def __repr__(self) -> str:
        return f"Article(article_id={self.article_id}, category_id={self.category_id}, words_count={self.words_count})"

