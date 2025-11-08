"""
模型管理服务：保存和加载训练好的模型
"""

import os
import pickle
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
from pathlib import Path
from typing import Optional, Dict, Any

from ..utils.exceptions import ModelTrainingError
from ..utils.logger import logger
from ..utils.config import DATA_DIR


class ModelManager:
    """模型管理器：负责模型的保存和加载"""
    
    def __init__(self, model_dir: str = None):
        """
        初始化模型管理器
        
        Args:
            model_dir: 模型保存目录，默认为data/models/
        """
        if model_dir is None:
            self.model_dir = Path(DATA_DIR) / 'models'
        else:
            self.model_dir = Path(model_dir)
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model directory: {self.model_dir}")
    
    def save_itemcf_model(self, itemcf, model_name: str = 'itemcf_model') -> str:
        """
        保存ItemCF模型（优化版本：只保存共现矩阵，不保存相似度矩阵）
        
        Args:
            itemcf: ItemCF模型实例
            model_name: 模型名称
            
        Returns:
            保存的文件路径
        """
        try:
            model_path = self.model_dir / f"{model_name}.pkl"
            cooccurrence_path = self.model_dir / f"{model_name}_cooccurrence.npz"
            matrix_path = self.model_dir / f"{model_name}_matrix.npz"
            
            # 保存物品共现矩阵（稀疏矩阵）
            if itemcf.item_cooccurrence is not None:
                save_npz(cooccurrence_path, itemcf.item_cooccurrence)
                logger.info(f"Saved ItemCF co-occurrence matrix to {cooccurrence_path}")
            
            # 保存用户-物品交互矩阵（用于按需计算相似度）
            if itemcf.user_item_matrix is not None:
                save_npz(matrix_path, itemcf.user_item_matrix)
                logger.info(f"Saved ItemCF user-item matrix to {matrix_path}")
            
            # 保存模型元数据
            model_data = {
                'article_ids': itemcf.article_ids,
                'article_id_to_index': itemcf.article_id_to_index,
                'cooccurrence_path': str(cooccurrence_path),
                'matrix_path': str(matrix_path)
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Saved ItemCF model to {model_path}")
            return str(model_path)
            
        except Exception as e:
            raise ModelTrainingError(f"Error saving ItemCF model: {str(e)}")
    
    def load_itemcf_model(self, model_name: str = 'itemcf_model'):
        """
        加载ItemCF模型（优化版本：只加载共现矩阵，不加载相似度矩阵）
        
        Args:
            model_name: 模型名称
            
        Returns:
            ItemCF模型实例
        """
        try:
            from ..algorithms.itemcf import ItemCF
            
            model_path = self.model_dir / f"{model_name}.pkl"
            if not model_path.exists():
                return None
            
            # 加载模型元数据
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # 加载物品共现矩阵（兼容旧版本：如果存在相似度矩阵路径，也尝试加载）
            cooccurrence_path = Path(model_data.get('cooccurrence_path', ''))
            if not cooccurrence_path.exists():
                # 兼容旧版本：尝试加载相似度矩阵路径
                similarity_path = Path(model_data.get('similarity_path', ''))
                if similarity_path.exists():
                    logger.warning("Found old format ItemCF model (similarity matrix), "
                                 "this will not be loaded. Please retrain the model.")
                return None
            
            cooccurrence_matrix = load_npz(cooccurrence_path)
            
            # 加载用户-物品交互矩阵
            matrix_path = Path(model_data.get('matrix_path', ''))
            if not matrix_path.exists():
                logger.warning("User-item matrix not found, ItemCF model may not work correctly")
                return None
            
            user_item_matrix = load_npz(matrix_path)
            
            # 重建模型
            itemcf = ItemCF()
            itemcf.article_ids = model_data['article_ids']
            itemcf.article_id_to_index = model_data['article_id_to_index']
            itemcf.item_cooccurrence = cooccurrence_matrix
            itemcf.user_item_matrix = user_item_matrix
            itemcf.item_similarity = None  # 不加载相似度矩阵
            
            logger.info(f"Loaded ItemCF model from {model_path}")
            return itemcf
            
        except Exception as e:
            logger.warning(f"Error loading ItemCF model: {str(e)}")
            return None
    
    def save_usercf_model(self, usercf, model_name: str = 'usercf_model') -> str:
        """
        保存UserCF模型
        
        Args:
            usercf: UserCF模型实例
            model_name: 模型名称
            
        Returns:
            保存的文件路径
        """
        try:
            model_path = self.model_dir / f"{model_name}.pkl"
            matrix_path = self.model_dir / f"{model_name}_matrix.npz"
            
            # 保存用户-物品交互矩阵（稀疏矩阵）
            if usercf.user_item_matrix is not None:
                save_npz(matrix_path, usercf.user_item_matrix)
                logger.info(f"Saved UserCF user-item matrix to {matrix_path}")
            
            # 保存模型元数据
            model_data = {
                'user_ids': usercf.user_ids,
                'user_id_to_index': usercf.user_id_to_index,
                'article_ids': usercf.article_ids,
                'matrix_path': str(matrix_path)
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Saved UserCF model to {model_path}")
            return str(model_path)
            
        except Exception as e:
            raise ModelTrainingError(f"Error saving UserCF model: {str(e)}")
    
    def load_usercf_model(self, model_name: str = 'usercf_model'):
        """
        加载UserCF模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            UserCF模型实例
        """
        try:
            from ..algorithms.usercf import UserCF
            
            model_path = self.model_dir / f"{model_name}.pkl"
            if not model_path.exists():
                return None
            
            # 加载模型元数据
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # 加载用户-物品交互矩阵
            matrix_path = Path(model_data['matrix_path'])
            if matrix_path.exists():
                user_item_matrix = load_npz(matrix_path)
            else:
                return None
            
            # 重建模型
            usercf = UserCF()
            usercf.user_ids = model_data['user_ids']
            usercf.user_id_to_index = model_data['user_id_to_index']
            usercf.article_ids = model_data['article_ids']
            usercf.user_item_matrix = user_item_matrix
            
            logger.info(f"Loaded UserCF model from {model_path}")
            return usercf
            
        except Exception as e:
            logger.warning(f"Error loading UserCF model: {str(e)}")
            return None
    
    def save_embedding_model(self, embedding_rec, model_name: str = 'embedding_model') -> str:
        """
        保存Embedding模型（优化版本：只保存embedding向量，不保存相似度矩阵）
        
        Args:
            embedding_rec: EmbeddingRecommender模型实例
            model_name: 模型名称
            
        Returns:
            保存的文件路径
        """
        try:
            model_path = self.model_dir / f"{model_name}.pkl"
            embeddings_path = self.model_dir / f"{model_name}_embeddings.npy"
            
            # 保存embedding向量（不保存相似度矩阵以节省空间）
            if embedding_rec.article_embeddings is not None:
                np.save(embeddings_path, embedding_rec.article_embeddings)
                logger.info(f"Saved article embeddings to {embeddings_path}")
            
            # 保存模型元数据
            model_data = {
                'article_ids': embedding_rec.article_ids,
                'article_id_to_index': embedding_rec.article_id_to_index,
                'embeddings_path': str(embeddings_path)
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Saved embedding model to {model_path}")
            return str(model_path)
            
        except Exception as e:
            raise ModelTrainingError(f"Error saving embedding model: {str(e)}")
    
    def load_embedding_model(self, model_name: str = 'embedding_model'):
        """
        加载Embedding模型（优化版本：只加载embedding向量）
        
        Args:
            model_name: 模型名称
            
        Returns:
            EmbeddingRecommender模型实例
        """
        try:
            from ..algorithms.embedding_based import EmbeddingRecommender
            
            model_path = self.model_dir / f"{model_name}.pkl"
            if not model_path.exists():
                return None
            
            # 加载模型元数据
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # 加载embedding向量（兼容旧版本：如果存在相似度矩阵路径，也尝试加载）
            embeddings_path = Path(model_data.get('embeddings_path', ''))
            if not embeddings_path.exists():
                # 兼容旧版本：尝试加载相似度矩阵路径
                similarity_path = Path(model_data.get('similarity_path', ''))
                if similarity_path.exists():
                    logger.warning("Found old format embedding model (similarity matrix), "
                                 "this will not be loaded. Please retrain the model.")
                return None
            
            embeddings = np.load(embeddings_path)
            
            # 重建模型
            embedding_rec = EmbeddingRecommender()
            embedding_rec.article_ids = model_data['article_ids']
            embedding_rec.article_id_to_index = model_data['article_id_to_index']
            embedding_rec.article_embeddings = embeddings.astype(np.float32)
            embedding_rec.embedding_similarity = None  # 不加载相似度矩阵
            
            logger.info(f"Loaded embedding model from {model_path}")
            return embedding_rec
            
        except Exception as e:
            logger.warning(f"Error loading embedding model: {str(e)}")
            return None
    
    def save_all_models(self, itemcf, usercf, embedding_rec, popular_articles: list = None) -> Dict[str, str]:
        """
        保存所有模型
        
        Args:
            itemcf: ItemCF模型实例
            usercf: UserCF模型实例
            embedding_rec: EmbeddingRecommender模型实例
            popular_articles: 热门文章列表（可选）
            
        Returns:
            保存的文件路径字典
        """
        paths = {}
        
        # 保存各个模型
        paths['itemcf'] = self.save_itemcf_model(itemcf)
        paths['usercf'] = self.save_usercf_model(usercf)
        paths['embedding'] = self.save_embedding_model(embedding_rec)
        
        # 保存热门文章列表
        if popular_articles is not None:
            popular_path = self.model_dir / 'popular_articles.pkl'
            with open(popular_path, 'wb') as f:
                pickle.dump(popular_articles, f)
            paths['popular_articles'] = str(popular_path)
            logger.info(f"Saved popular articles to {popular_path}")
        
        return paths
    
    def load_all_models(self) -> Optional[Dict[str, Any]]:
        """
        加载所有模型
        
        Returns:
            包含所有模型的字典，如果加载失败返回None
        """
        itemcf = self.load_itemcf_model()
        usercf = self.load_usercf_model()
        embedding_rec = self.load_embedding_model()
        
        if itemcf is None or usercf is None or embedding_rec is None:
            return None
        
        # 加载热门文章列表
        popular_path = self.model_dir / 'popular_articles.pkl'
        popular_articles = None
        if popular_path.exists():
            with open(popular_path, 'rb') as f:
                popular_articles = pickle.load(f)
        
        return {
            'itemcf': itemcf,
            'usercf': usercf,
            'embedding': embedding_rec,
            'popular_articles': popular_articles
        }
    
    def model_exists(self, model_name: str) -> bool:
        """
        检查模型是否存在
        
        Args:
            model_name: 模型名称（itemcf/usercf/embedding）
            
        Returns:
            模型是否存在
        """
        model_path = self.model_dir / f"{model_name}_model.pkl"
        return model_path.exists()

