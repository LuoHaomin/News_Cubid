"""
基于Embedding的推荐算法
"""

import numpy as np
from typing import List

from ..utils.exceptions import ModelTrainingError, RecommendationError
from ..utils.similarity import cosine_similarity_dense, validate_similarity_matrix
from ..utils.logger import logger


class EmbeddingRecommender:
    """基于Embedding的推荐算法"""
    
    def __init__(self):
        """初始化Embedding推荐算法"""
        self.article_embeddings = None
        self.article_ids = None
        self.article_id_to_index = None
        self.embedding_similarity = None
    
    def compute_similarity(self, embeddings: np.ndarray, article_ids: List[str] = None) -> None:
        """
        加载文章embedding向量（优化版本：不预先计算完整相似度矩阵）
        
        注意：为了节省内存，本方法不再计算完整的文章相似度矩阵。
        相似度将在推荐时按需计算（用户embedding与所有文章embedding的相似度）。
        
        Args:
            embeddings: 文章embedding向量矩阵，形状为(n_articles, dim)
            article_ids: 文章ID列表（可选）
            
        Returns:
            None（不再返回相似度矩阵）
            
        Raises:
            ModelTrainingError: 模型训练错误
        """
        try:
            logger.info("Loading article embeddings (optimized: on-demand similarity computation)...")
            
            # 自动检测embedding维度
            actual_dim = embeddings.shape[1]
            if actual_dim <= 0:
                raise ModelTrainingError(f"Invalid embedding dimension: {actual_dim}")
            logger.info(f"Using embedding dimension: {actual_dim}, {len(embeddings)} articles")
            
            # 保存embedding向量（使用float32以节省内存）
            self.article_embeddings = embeddings.astype(np.float32)
            if article_ids is not None:
                self.article_ids = article_ids
                self.article_id_to_index = {aid: idx for idx, aid in enumerate(article_ids)}
            
            # 不预先计算完整的文章相似度矩阵（内存优化）
            # 相似度将在推荐时按需计算（用户embedding与所有文章embedding的相似度）
            self.embedding_similarity = None
            
            logger.info(f"Article embeddings loaded: {len(embeddings)} articles with dimension {actual_dim}")
            logger.info("Note: Similarity will be computed on-demand during recommendation to save memory")
            
        except Exception as e:
            raise ModelTrainingError(f"Error loading article embeddings: {str(e)}")
    
    def recommend(self,
                  user_id: str,
                  user_history: List[str],
                  article_embeddings: np.ndarray = None,
                  article_ids: List[str] = None,
                  top_k: int = 5) -> List[str]:
        """
        使用embedding相似度为用户生成推荐
        
        Args:
            user_id: 用户ID
            user_history: 用户历史点击文章ID列表
            article_embeddings: 文章embedding向量矩阵（可选）
            article_ids: 所有文章ID列表（可选）
            top_k: 推荐数量，默认5
            
        Returns:
            推荐文章ID列表（长度为top_k）
            
        Raises:
            RecommendationError: 推荐生成错误
        """
        try:
            if article_embeddings is None:
                article_embeddings = self.article_embeddings
            if article_ids is None:
                article_ids = self.article_ids
            
            if article_embeddings is None or article_ids is None:
                raise RecommendationError("Embeddings not loaded. Call compute_similarity() first or provide article_embeddings and article_ids.")
            
            if not user_history:
                # 冷启动：返回空列表，由上层处理
                return []
            
            # 获取用户历史点击的文章embedding
            user_embeddings = []
            clicked_indices = []
            for article_id in user_history:
                if article_id in self.article_id_to_index:
                    idx = self.article_id_to_index[article_id]
                    user_embeddings.append(article_embeddings[idx])
                    clicked_indices.append(idx)
            
            if not user_embeddings:
                return []
            
            # 计算用户embedding（平均用户历史点击文章的embedding）
            user_embedding = np.mean(user_embeddings, axis=0)
            
            # 计算用户embedding与所有文章embedding的相似度
            # 归一化用户embedding
            user_norm = np.linalg.norm(user_embedding)
            if user_norm > 0:
                user_embedding = user_embedding / user_norm
            
            # 归一化文章embedding
            article_norms = np.linalg.norm(article_embeddings, axis=1, keepdims=True)
            article_norms[article_norms == 0] = 1
            normalized_embeddings = article_embeddings / article_norms
            
            # 计算余弦相似度
            scores = np.dot(normalized_embeddings, user_embedding)
            
            # 归一化到[0, 1]
            scores = (scores + 1) / 2
            
            # 排除用户已点击的文章
            for clicked_idx in clicked_indices:
                scores[clicked_idx] = 0.0
            
            # 选择top_k个文章（确保所有文章ID都是字符串类型）
            top_indices = np.argsort(scores)[::-1][:top_k]
            recommended_articles = [str(article_ids[idx]) for idx in top_indices if scores[idx] > 0 and idx < len(article_ids)]
            
            # 如果推荐不足top_k，补充
            if len(recommended_articles) < top_k:
                user_history_str = {str(aid) for aid in user_history}
                remaining_articles = [str(aid) for aid in article_ids if str(aid) not in user_history_str and str(aid) not in recommended_articles]
                needed = top_k - len(recommended_articles)
                if len(remaining_articles) >= needed:
                    import random
                    recommended_articles.extend(random.sample(remaining_articles, needed))
                else:
                    recommended_articles.extend(remaining_articles)
            
            return recommended_articles[:top_k]
            
        except Exception as e:
            raise RecommendationError(f"Error generating embedding-based recommendations: {str(e)}")

