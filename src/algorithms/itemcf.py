"""
ItemCF协同过滤算法
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Dict

from ..utils.exceptions import ModelTrainingError, RecommendationError
from ..utils.similarity import cosine_similarity_sparse, validate_similarity_matrix
from ..utils.logger import logger


class ItemCF:
    """ItemCF协同过滤推荐算法"""
    
    def __init__(self):
        """初始化ItemCF算法"""
        self.item_similarity = None  # 不再保存完整相似度矩阵
        self.item_cooccurrence = None  # 保存物品共现矩阵（用于按需计算相似度）
        self.user_item_matrix = None  # 保存用户-物品交互矩阵（用于按需计算相似度）
        self.article_ids = None
        self.article_id_to_index = None
    
    def train(self, click_log: pd.DataFrame, articles: pd.DataFrame, 
              user_item_matrix: csr_matrix = None,
              user_id_to_index: dict = None,
              article_id_to_index: dict = None) -> csr_matrix:
        """
        训练ItemCF协同过滤模型
        
        Args:
            click_log: 用户点击日志
            articles: 文章信息
            user_item_matrix: 用户-物品交互矩阵（可选，如果提供则直接使用，避免重复计算）
            user_id_to_index: 用户ID到索引的映射（可选，与user_item_matrix一起提供）
            article_id_to_index: 文章ID到索引的映射（可选，与user_item_matrix一起提供）
            
        Returns:
            文章相似度矩阵（稀疏矩阵）
            
        Raises:
            ModelTrainingError: 模型训练错误
        """
        try:
            logger.info("Training ItemCF model...")
            
            # 获取所有文章ID
            self.article_ids = articles['article_id'].unique().tolist()
            self.article_id_to_index = {aid: idx for idx, aid in enumerate(self.article_ids)}
            n_articles = len(self.article_ids)
            logger.info(f"Found {n_articles} articles")
            
            # 如果提供了共享的user_item_matrix，直接使用
            if user_item_matrix is not None and user_id_to_index is not None and article_id_to_index is not None:
                logger.info("Using shared user-item interaction matrix...")
                # 需要将共享矩阵的列索引映射到ItemCF的文章索引
                # 因为ItemCF使用articles中的所有文章，而共享矩阵可能只包含click_log中的文章
                # 这里简化处理：如果共享矩阵的文章索引与ItemCF的文章索引一致，直接使用
                # 否则需要重新构建
                if set(article_id_to_index.keys()) == set(self.article_id_to_index.keys()):
                    user_item_matrix_shared = user_item_matrix
                else:
                    logger.info("Article index mismatch, rebuilding matrix for ItemCF...")
                    user_item_matrix_shared = None
            else:
                user_item_matrix_shared = None
            
            # 如果没有共享矩阵或索引不匹配，则构建新的矩阵
            if user_item_matrix_shared is None:
                # 构建用户-物品交互矩阵
                user_ids = click_log['user_id'].unique()
                user_id_to_index_local = {uid: idx for idx, uid in enumerate(user_ids)}
                n_users = len(user_ids)
                logger.info(f"Found {n_users} users")
                
                logger.info("Building user-item interaction matrix...")
                # 创建稀疏矩阵（用户-物品交互矩阵）
                rows = []
                cols = []
                data = []
                
                total_rows = len(click_log)
                for idx, (_, row) in enumerate(click_log.iterrows()):
                    if (idx + 1) % 100000 == 0:
                        logger.info(f"  Processing click logs: {idx + 1}/{total_rows} ({100*(idx+1)/total_rows:.1f}%)")
                    
                    user_idx = user_id_to_index_local[row['user_id']]
                    article_id = row['click_article_id']
                    if article_id in self.article_id_to_index:
                        article_idx = self.article_id_to_index[article_id]
                        rows.append(user_idx)
                        cols.append(article_idx)
                        data.append(1.0)  # 二进制交互
                
                logger.info("Creating sparse matrix...")
                user_item_matrix_shared = csr_matrix((data, (rows, cols)), shape=(n_users, n_articles))
                logger.info(f"User-item matrix created: shape {user_item_matrix_shared.shape}, "
                           f"non-zero elements: {user_item_matrix_shared.nnz}")
            
            # 计算物品共现矩阵（物品-物品共现）
            logger.info("Computing item co-occurrence matrix...")
            item_cooccurrence = user_item_matrix_shared.T.dot(user_item_matrix_shared)
            logger.info(f"Item co-occurrence matrix created: shape {item_cooccurrence.shape}, "
                       f"non-zero elements: {item_cooccurrence.nnz}")
            
            # 不预先计算完整的物品相似度矩阵（内存优化）
            # 相似度将在推荐时按需计算
            self.item_cooccurrence = item_cooccurrence
            self.user_item_matrix = user_item_matrix_shared
            self.item_similarity = None  # 不保存完整相似度矩阵
            
            logger.info(f"ItemCF model trained successfully: {n_articles} articles, "
                       f"co-occurrence matrix shape {item_cooccurrence.shape}")
            logger.info("Note: Item similarity will be computed on-demand during recommendation to save memory")
            return item_cooccurrence  # 返回共现矩阵而不是相似度矩阵
            
        except Exception as e:
            raise ModelTrainingError(f"Error training ItemCF model: {str(e)}")
    
    def _compute_item_similarity_on_demand(self, item_idx: int) -> np.ndarray:
        """
        按需计算物品相似度（内存优化版本：只计算有共现的物品）
        
        Args:
            item_idx: 物品索引
            
        Returns:
            相似度分数数组（对所有物品）
        """
        if self.item_cooccurrence is None or self.user_item_matrix is None:
            return np.zeros(len(self.article_ids)) if self.article_ids else np.array([])
        
        # 获取当前物品的共现信息（从共现矩阵中提取）
        cooccurrence_row = self.item_cooccurrence[item_idx, :]
        
        # 获取有共现的物品索引
        cooccurred_indices = cooccurrence_row.indices
        
        if len(cooccurred_indices) == 0:
            return np.zeros(len(self.article_ids))
        
        # 计算当前物品的向量（所有用户对物品的交互）
        item_vector = self.user_item_matrix[:, item_idx]
        item_norm = np.sqrt(item_vector.multiply(item_vector).sum())
        if item_norm == 0:
            return np.zeros(len(self.article_ids))
        
        # 归一化当前物品向量
        normalized_item = item_vector.multiply(1.0 / item_norm)
        
        # 计算相似度（只计算有共现的物品）
        similarities = np.zeros(len(self.article_ids))
        
        # 分块计算相似度（避免一次性加载所有数据）
        chunk_size = 1000
        for i in range(0, len(cooccurred_indices), chunk_size):
            end_idx = min(i + chunk_size, len(cooccurred_indices))
            chunk_indices = cooccurred_indices[i:end_idx]
            
            # 获取共现物品的向量
            cooccurred_vectors = self.user_item_matrix[:, chunk_indices]
            
            # 计算每个物品的范数
            cooccurred_norms = np.sqrt(cooccurred_vectors.multiply(cooccurred_vectors).sum(axis=0))
            cooccurred_norms = np.array(cooccurred_norms).flatten()
            cooccurred_norms[cooccurred_norms == 0] = 1
            
            # 归一化
            normalized_cooccurred = cooccurred_vectors.multiply(1.0 / cooccurred_norms[:, np.newaxis])
            
            # 计算余弦相似度
            chunk_similarities_dot = normalized_cooccurred.T.dot(normalized_item)
            # 处理可能是稀疏矩阵或numpy matrix的情况
            if hasattr(chunk_similarities_dot, 'toarray'):
                chunk_similarities = chunk_similarities_dot.toarray().flatten()
            elif hasattr(chunk_similarities_dot, 'A'):
                chunk_similarities = chunk_similarities_dot.A.flatten()
            else:
                chunk_similarities = np.array(chunk_similarities_dot).flatten()
            
            # 归一化到[0, 1]
            chunk_similarities = np.clip((chunk_similarities + 1) / 2, 0.0, 1.0)
            
            similarities[chunk_indices] = chunk_similarities
        
        return similarities
    
    def recommend(self, 
                  user_id: str,
                  user_history: List[str],
                  item_similarity: csr_matrix = None,
                  article_ids: List[str] = None,
                  top_k: int = 5) -> List[str]:
        """
        使用ItemCF算法为用户生成推荐（优化版本：按需计算相似度）
        
        Args:
            user_id: 用户ID
            user_history: 用户历史点击文章ID列表
            item_similarity: 文章相似度矩阵（可选，已废弃，使用按需计算）
            article_ids: 所有文章ID列表（可选）
            top_k: 推荐数量，默认5
            
        Returns:
            推荐文章ID列表（长度为top_k）
            
        Raises:
            RecommendationError: 推荐生成错误
        """
        try:
            if article_ids is None:
                article_ids = self.article_ids
            
            if self.item_cooccurrence is None or self.user_item_matrix is None or article_ids is None:
                raise RecommendationError("Model not trained. Call train() first.")
            
            # 获取用户历史点击的文章索引
            clicked_indices = []
            for article_id in user_history:
                if article_id in self.article_id_to_index:
                    clicked_indices.append(self.article_id_to_index[article_id])
            
            if not clicked_indices:
                # 冷启动：返回空列表，由上层处理
                return []
            
            # 计算推荐分数：对用户点击过的所有文章，累加它们的相似文章
            scores = np.zeros(len(article_ids))
            for clicked_idx in clicked_indices:
                # 按需计算与当前文章相似的所有文章
                similarities = self._compute_item_similarity_on_demand(clicked_idx)
                scores += similarities
            
            # 排除用户已点击的文章
            for clicked_idx in clicked_indices:
                scores[clicked_idx] = 0.0
            
            # 选择top_k个文章
            top_indices = np.argsort(scores)[::-1][:top_k]
            recommended_articles = [str(article_ids[idx]) for idx in top_indices if scores[idx] > 0 and idx < len(article_ids)]
            
            # 如果推荐不足top_k，用热门文章补充（这里简化处理，实际应该从训练数据中获取热门文章）
            if len(recommended_articles) < top_k:
                # 从所有文章中随机选择补充（实际应该选择热门文章）
                remaining_articles = [str(aid) for aid in article_ids if str(aid) not in user_history and str(aid) not in recommended_articles]
                needed = top_k - len(recommended_articles)
                if len(remaining_articles) >= needed:
                    import random
                    recommended_articles.extend(random.sample(remaining_articles, needed))
                else:
                    recommended_articles.extend(remaining_articles)
            
            return recommended_articles[:top_k]
            
        except Exception as e:
            raise RecommendationError(f"Error generating ItemCF recommendations: {str(e)}")

