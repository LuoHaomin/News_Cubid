"""
推荐结果模型
"""

from typing import List, Optional


class Recommendation:
    """推荐结果实体类"""
    
    def __init__(self,
                 user_id: str,
                 article_ids: List[str],
                 scores: Optional[List[float]] = None,
                 algorithm: Optional[str] = None):
        """
        初始化推荐结果
        
        Args:
            user_id: 用户ID
            article_ids: 推荐文章ID列表（5篇）
            scores: 推荐分数列表（可选）
            algorithm: 使用的推荐算法（ItemCF/UserCF/Embedding）
        """
        self.user_id = user_id
        self.article_ids = article_ids
        self.scores = scores
        self.algorithm = algorithm
    
    def __repr__(self) -> str:
        return f"Recommendation(user_id={self.user_id}, article_count={len(self.article_ids)}, algorithm={self.algorithm})"

