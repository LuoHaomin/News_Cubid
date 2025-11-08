"""
UserCF协同过滤算法
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Dict

from ..utils.exceptions import ModelTrainingError, RecommendationError
from ..utils.similarity import cosine_similarity_sparse, validate_similarity_matrix
from ..utils.logger import logger


class UserCF:
    """UserCF协同过滤推荐算法"""
    
    def __init__(self):
        """初始化UserCF算法"""
        self.user_similarity = None
        self.user_ids = None
        self.user_id_to_index = None
        self.user_item_matrix = None
        self.article_ids = None
    
    def train(self, click_log: pd.DataFrame,
              user_item_matrix: csr_matrix = None,
              user_id_to_index: dict = None,
              article_id_to_index: dict = None) -> csr_matrix:
        """
        训练UserCF协同过滤模型（优化版本：不预先计算完整相似度矩阵）
        
        Args:
            click_log: 用户点击日志
            user_item_matrix: 用户-物品交互矩阵（可选，如果提供则直接使用，避免重复计算）
            user_id_to_index: 用户ID到索引的映射（可选，与user_item_matrix一起提供）
            article_id_to_index: 文章ID到索引的映射（可选，与user_item_matrix一起提供）
            
        Returns:
            用户-物品交互矩阵（用于按需计算相似度）
            
        Raises:
            ModelTrainingError: 模型训练错误
        """
        try:
            logger.info("Training UserCF model (optimized: on-demand similarity computation)...")
            
            # 如果提供了共享的user_item_matrix，直接使用
            if user_item_matrix is not None and user_id_to_index is not None and article_id_to_index is not None:
                logger.info("Using shared user-item interaction matrix...")
                self.user_item_matrix = user_item_matrix
                self.user_ids = list(user_id_to_index.keys())
                self.article_ids = list(article_id_to_index.keys())
                self.user_id_to_index = user_id_to_index
                n_users = len(self.user_ids)  # 定义n_users用于日志输出
                logger.info(f"User-item matrix loaded: shape {self.user_item_matrix.shape}, "
                           f"non-zero elements: {self.user_item_matrix.nnz}")
            else:
                # 获取所有用户ID和文章ID
                self.user_ids = click_log['user_id'].unique().tolist()
                self.article_ids = click_log['click_article_id'].unique().tolist()
                
                self.user_id_to_index = {uid: idx for idx, uid in enumerate(self.user_ids)}
                article_id_to_index = {aid: idx for idx, aid in enumerate(self.article_ids)}
                
                n_users = len(self.user_ids)
                n_articles = len(self.article_ids)
                
                logger.info(f"Building user-item matrix: {n_users} users, {n_articles} articles")
                
                # 构建用户-物品交互矩阵
                rows = []
                cols = []
                data = []
                
                total_rows = len(click_log)
                for idx, (_, row) in enumerate(click_log.iterrows()):
                    if (idx + 1) % 100000 == 0:
                        logger.info(f"  Processing click logs: {idx + 1}/{total_rows} ({100*(idx+1)/total_rows:.1f}%)")
                    
                    user_idx = self.user_id_to_index[row['user_id']]
                    article_id = row['click_article_id']
                    if article_id in article_id_to_index:
                        article_idx = article_id_to_index[article_id]
                        rows.append(user_idx)
                        cols.append(article_idx)
                        data.append(1.0)  # 二进制交互
                
                logger.info("Creating sparse matrix...")
                self.user_item_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_articles), dtype=np.float32)
                logger.info(f"User-item matrix created: shape {self.user_item_matrix.shape}, "
                           f"non-zero elements: {self.user_item_matrix.nnz}")
            
            # 不预先计算完整的用户相似度矩阵（内存优化）
            # 相似度将在推荐时按需计算
            self.user_similarity = None
            
            logger.info(f"UserCF model trained: {n_users} users, user-item matrix shape {self.user_item_matrix.shape}")
            logger.info("Note: User similarity will be computed on-demand during recommendation to save memory")
            return self.user_item_matrix
            
        except Exception as e:
            raise ModelTrainingError(f"Error training UserCF model: {str(e)}")
    
    def _compute_user_similarity_on_demand(self, user_idx: int, top_k_similar: int = 50) -> np.ndarray:
        """
        按需计算用户相似度（内存优化版本：只计算有共同交互的用户）
        
        Args:
            user_idx: 用户索引
            top_k_similar: 返回前k个相似用户，默认50
            
        Returns:
            相似用户索引数组
        """
        if self.user_item_matrix is None:
            return np.array([])
        
        # 获取当前用户的向量（稀疏向量）
        user_vector = self.user_item_matrix[user_idx, :]
        
        # 计算用户向量的范数
        user_norm = np.sqrt(user_vector.multiply(user_vector).sum())
        if user_norm == 0:
            return np.array([])
        
        # 获取当前用户交互过的文章索引
        user_items = user_vector.indices
        
        if len(user_items) == 0:
            return np.array([])
        
        # 只计算与当前用户有共同交互的用户（内存优化）
        # 使用更高效的方法：直接计算交集，避免创建大型矩阵
        # 获取所有用户在这些文章上的交互（只获取非零部分）
        n_users = self.user_item_matrix.shape[0]
        
        # 使用稀疏矩阵的列切片（只获取需要的列）
        # 但为了避免内存问题，我们使用更高效的方法
        # 直接遍历用户，只计算有共同交互的用户
        candidate_users = []
        
        # 分块处理，避免一次性加载所有数据
        chunk_size = 5000
        for i in range(0, n_users, chunk_size):
            end_idx = min(i + chunk_size, n_users)
            chunk = self.user_item_matrix[i:end_idx, :]
            
            # 计算每个用户与当前用户的交集
            chunk_intersection = chunk[:, user_items].sum(axis=1)
            # 处理可能是稀疏矩阵或numpy matrix的情况
            if hasattr(chunk_intersection, 'toarray'):
                chunk_intersection = chunk_intersection.toarray().flatten()
            elif hasattr(chunk_intersection, 'A'):
                chunk_intersection = chunk_intersection.A.flatten()
            else:
                chunk_intersection = np.array(chunk_intersection).flatten()
            
            # 找到有共同交互的用户
            chunk_candidates = np.where(chunk_intersection > 0)[0] + i
            chunk_candidates = chunk_candidates[chunk_candidates != user_idx]  # 排除自己
            candidate_users.extend(chunk_candidates.tolist())
        
        candidate_users = np.array(candidate_users, dtype=np.int32)
        
        if len(candidate_users) == 0:
            return np.array([])
        
        # 计算候选用户的相似度（余弦相似度）
        # 只计算候选用户，大大减少计算量
        user_similarities = np.zeros(len(candidate_users), dtype=np.float32)
        
        # 归一化当前用户向量
        normalized_user = user_vector.multiply(1.0 / user_norm)
        
        # 分块计算候选用户的相似度
        chunk_size = 1000
        for i in range(0, len(candidate_users), chunk_size):
            end_idx = min(i + chunk_size, len(candidate_users))
            chunk_indices = candidate_users[i:end_idx]
            chunk_matrix = self.user_item_matrix[chunk_indices, :]
            
            # 计算每个用户的范数
            chunk_norms = np.sqrt(chunk_matrix.multiply(chunk_matrix).sum(axis=1))
            chunk_norms = np.array(chunk_norms).flatten()
            chunk_norms[chunk_norms == 0] = 1
            
            # 归一化
            normalized_chunk = chunk_matrix.multiply(1.0 / chunk_norms[:, np.newaxis])
            
            # 计算余弦相似度
            chunk_similarities_dot = normalized_chunk.dot(normalized_user.T)
            # 处理可能是稀疏矩阵或numpy matrix的情况
            if hasattr(chunk_similarities_dot, 'toarray'):
                chunk_similarities = chunk_similarities_dot.toarray().flatten()
            elif hasattr(chunk_similarities_dot, 'A'):
                chunk_similarities = chunk_similarities_dot.A.flatten()
            else:
                chunk_similarities = np.array(chunk_similarities_dot).flatten()
            
            # 归一化到[0, 1]
            chunk_similarities = np.clip((chunk_similarities + 1) / 2, 0.0, 1.0)
            
            user_similarities[i:end_idx] = chunk_similarities
        
        # 获取top-k相似用户
        top_k = min(top_k_similar, len(candidate_users))
        top_indices_in_candidates = np.argsort(user_similarities)[::-1][:top_k]
        top_user_indices = candidate_users[top_indices_in_candidates]
        
        return top_user_indices
    
    def recommend(self,
                  user_id: str,
                  user_similarity: csr_matrix = None,
                  user_ids: List[str] = None,
                  click_log: pd.DataFrame = None,
                  top_k: int = 5) -> List[str]:
        """
        使用UserCF算法为用户生成推荐（优化版本：按需计算相似度）
        
        Args:
            user_id: 用户ID
            user_similarity: 用户相似度矩阵（可选，已废弃，使用按需计算）
            user_ids: 所有用户ID列表（可选）
            click_log: 用户点击日志（必需）
            top_k: 推荐数量，默认5
            
        Returns:
            推荐文章ID列表（长度为top_k）
            
        Raises:
            RecommendationError: 推荐生成错误
        """
        try:
            if click_log is None:
                raise RecommendationError("click_log is required for UserCF recommendation")
            
            if user_id not in self.user_id_to_index:
                # 冷启动用户
                return []
            
            if self.user_item_matrix is None:
                raise RecommendationError("Model not trained. Call train() first.")
            
            user_idx = self.user_id_to_index[user_id]
            
            # 按需计算相似用户（只计算top-50相似用户以节省内存）
            top_similar_user_indices = self._compute_user_similarity_on_demand(user_idx, top_k_similar=50)
            
            if len(top_similar_user_indices) == 0:
                return []
            
            # 获取用户已点击的文章（确保所有文章ID都是字符串类型）
            user_clicked = {str(aid) for aid in click_log[click_log['user_id'] == user_id]['click_article_id'].tolist()}
            
            # 计算相似度分数（简化版本，使用共现次数作为权重）
            # 从相似用户点击的文章中推荐
            article_scores = {}
            for similar_user_idx in top_similar_user_indices:
                similar_user_id = self.user_ids[similar_user_idx]
                similar_user_articles = [str(aid) for aid in click_log[click_log['user_id'] == similar_user_id]['click_article_id'].tolist()]
                for article_id in similar_user_articles:
                    if article_id not in user_clicked:
                        # 使用简单的共现计数作为分数
                        article_scores[article_id] = article_scores.get(article_id, 0) + 1
            
            # 按分数排序，选择top_k（确保所有文章ID都是字符串类型）
            sorted_articles = sorted(article_scores.items(), key=lambda x: x[1], reverse=True)
            recommended_articles = [str(aid) for aid, score in sorted_articles[:top_k] if aid is not None]
            
            # 如果推荐不足top_k，补充
            if len(recommended_articles) < top_k:
                all_articles = [str(aid) for aid in click_log['click_article_id'].unique().tolist()]
                user_clicked_str = {str(aid) for aid in user_clicked}
                remaining_articles = [aid for aid in all_articles if aid not in user_clicked_str and aid not in recommended_articles]
                needed = top_k - len(recommended_articles)
                if len(remaining_articles) >= needed:
                    import random
                    recommended_articles.extend(random.sample(remaining_articles, needed))
                else:
                    recommended_articles.extend(remaining_articles)
            
            return recommended_articles[:top_k]
            
        except Exception as e:
            raise RecommendationError(f"Error generating UserCF recommendations: {str(e)}")

