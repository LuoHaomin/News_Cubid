"""
推荐服务模块
"""

from typing import List, Dict
import pandas as pd
import numpy as np

from ..algorithms.itemcf import ItemCF
from ..algorithms.usercf import UserCF
from ..algorithms.embedding_based import EmbeddingRecommender
from ..utils.exceptions import RecommendationError
from ..utils.logger import logger
from ..utils.config import TOP_K


class BaselineRecommender:
    """基线推荐系统主类"""
    
    def __init__(self):
        """初始化推荐系统"""
        self.itemcf = ItemCF()
        self.usercf = UserCF()
        self.embedding_rec = EmbeddingRecommender()
        self.popular_articles = None
    
    def merge_recommendations(self,
                             itemcf_recs: List[str],
                             usercf_recs: List[str],
                             embedding_recs: List[str],
                             top_k: int = TOP_K) -> List[str]:
        """
        融合多路召回结果
        
        Args:
            itemcf_recs: ItemCF推荐结果
            usercf_recs: UserCF推荐结果
            embedding_recs: Embedding推荐结果
            top_k: 最终推荐数量，默认5
            
        Returns:
            融合后的推荐文章ID列表（长度为top_k，字符串类型）
        """
        # 统计每个文章出现的次数（确保所有文章ID都是字符串类型）
        article_counts = {}
        for article_id in itemcf_recs:
            if article_id is not None:
                aid_str = str(article_id).strip()
                if aid_str:
                    article_counts[aid_str] = article_counts.get(aid_str, 0) + 1
        for article_id in usercf_recs:
            if article_id is not None:
                aid_str = str(article_id).strip()
                if aid_str:
                    article_counts[aid_str] = article_counts.get(aid_str, 0) + 1
        for article_id in embedding_recs:
            if article_id is not None:
                aid_str = str(article_id).strip()
                if aid_str:
                    article_counts[aid_str] = article_counts.get(aid_str, 0) + 1
        
        # 按出现次数排序
        sorted_articles = sorted(article_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 取前top_k个
        merged_recs = [aid for aid, count in sorted_articles[:top_k]]
        
        # 如果不足top_k，用热门文章补充
        if len(merged_recs) < top_k and self.popular_articles:
            needed = top_k - len(merged_recs)
            remaining = [aid for aid in self.popular_articles if aid not in merged_recs]
            merged_recs.extend(remaining[:needed])
        
        return merged_recs[:top_k]
    
    def get_popular_articles(self, click_log: pd.DataFrame, top_n: int = 100) -> List[str]:
        """
        获取热门文章列表
        
        Args:
            click_log: 用户点击日志
            top_n: 返回前N篇热门文章
            
        Returns:
            热门文章ID列表（字符串类型）
        """
        article_counts = click_log['click_article_id'].value_counts()
        popular_articles = [str(aid) for aid in article_counts.head(top_n).index.tolist()]
        self.popular_articles = popular_articles
        return popular_articles
    
    def recommend_for_user(self,
                          user_id: str,
                          user_history: List[str],
                          train_click_log: pd.DataFrame,
                          top_k: int = TOP_K) -> List[str]:
        """
        为用户生成推荐（多路召回融合）
        
        Args:
            user_id: 用户ID
            user_history: 用户历史点击文章ID列表
            train_click_log: 训练集点击日志
            top_k: 推荐数量，默认5
            
        Returns:
            推荐文章ID列表
        """
        try:
            # ItemCF推荐
            try:
                itemcf_recs = self.itemcf.recommend(user_id, user_history, top_k=top_k)
                if len(itemcf_recs) == 0:
                    logger.debug(f"ItemCF returned empty recommendations for user {user_id} (history: {len(user_history)} articles)")
            except Exception as e:
                logger.warning(f"ItemCF recommendation failed for user {user_id}: {str(e)}")
                itemcf_recs = []
            
            # UserCF推荐
            try:
                usercf_recs = self.usercf.recommend(user_id, None, None, train_click_log, top_k=top_k)
                if len(usercf_recs) == 0:
                    logger.debug(f"UserCF returned empty recommendations for user {user_id}")
            except Exception as e:
                logger.warning(f"UserCF recommendation failed for user {user_id}: {str(e)}")
                usercf_recs = []
            
            # Embedding推荐
            try:
                embedding_recs = self.embedding_rec.recommend(user_id, user_history, top_k=top_k)
                if len(embedding_recs) == 0:
                    logger.debug(f"Embedding returned empty recommendations for user {user_id} (history: {len(user_history)} articles)")
            except Exception as e:
                logger.warning(f"Embedding recommendation failed for user {user_id}: {str(e)}")
                embedding_recs = []
            
            # 融合结果
            final_recs = self.merge_recommendations(itemcf_recs, usercf_recs, embedding_recs, top_k)
            
            # 过滤无效的文章ID（确保所有文章ID都是字符串类型）
            valid_article_ids = set(train_click_log['click_article_id'].astype(str).unique())
            user_history_str = {str(aid) for aid in user_history}  # 转换为字符串集合以便比较
            final_recs = [str(aid) for aid in final_recs if aid is not None and str(aid).strip() and str(aid) in valid_article_ids and str(aid) not in user_history_str]
            
            # 如果所有推荐算法都返回空，记录警告
            if len(itemcf_recs) == 0 and len(usercf_recs) == 0 and len(embedding_recs) == 0:
                logger.debug(f"All recommendation algorithms returned empty for user {user_id}, using popular articles")
            
            # 冷启动处理：如果推荐不足，使用热门文章（但要排除用户已点击的）
            if len(final_recs) < top_k:
                if self.popular_articles is None:
                    self.get_popular_articles(train_click_log)
                needed = top_k - len(final_recs)
                # 确保热门文章也是字符串类型，并排除用户已点击的
                popular_articles_str = [str(aid) for aid in self.popular_articles]
                remaining = [aid for aid in popular_articles_str if aid not in user_history_str and aid not in final_recs]
                final_recs.extend(remaining[:needed])
            
            # 确保返回恰好top_k个（如果还是不足，用随机文章补充，但要确保每个用户不同）
            if len(final_recs) < top_k:
                all_articles = [str(aid) for aid in train_click_log['click_article_id'].unique().tolist()]
                remaining = [aid for aid in all_articles if aid not in user_history_str and aid not in final_recs]
                if len(remaining) > 0:
                    import random
                    # 使用用户ID作为随机种子，确保每个用户得到不同的随机推荐
                    random.seed(hash(str(user_id)) % (2**32))
                    needed = min(top_k - len(final_recs), len(remaining))
                    if needed > 0:
                        final_recs.extend(random.sample(remaining, needed))
            
            return final_recs[:top_k]
            
        except Exception as e:
            raise RecommendationError(f"Error generating recommendations for user {user_id}: {str(e)}")

