"""
数据加载服务模块
"""

import os
import pandas as pd
import numpy as np
from typing import List
from pathlib import Path

from ..utils.exceptions import DataLoadError
from ..utils.validation import validate_dataframe_columns, validate_embedding
from ..utils.logger import logger


class DataLoader:
    """数据加载器类"""
    
    # 必需的列定义
    CLICK_LOG_COLUMNS = [
        'user_id', 'click_article_id', 'click_timestamp',
        'click_environment', 'click_deviceGroup', 'click_os',
        'click_country', 'click_region', 'click_referrer_type'
    ]
    
    ARTICLES_COLUMNS = ['article_id', 'category_id', 'created_at_ts', 'words_count']
    
    EMBEDDING_DIM = None  # 自动检测维度
    
    def load_train_data(self, file_path: str) -> pd.DataFrame:
        """
        加载训练集用户点击日志
        
        Args:
            file_path: 训练集CSV文件路径
            
        Returns:
            包含用户点击记录的DataFrame
            
        Raises:
            FileNotFoundError: 文件不存在
            DataLoadError: 数据加载错误
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Training data file not found: {file_path}")
            
            logger.info(f"Loading training data from {file_path}")
            logger.info("Reading CSV file (this may take a while for large files)...")
            df = pd.read_csv(file_path)
            logger.info(f"CSV file loaded: {len(df)} rows")
            
            # 验证必需的列
            logger.info("Validating data columns...")
            validate_dataframe_columns(df, self.CLICK_LOG_COLUMNS)
            
            # 数据验证
            if df.empty:
                raise DataLoadError("Training data is empty")
            
            # 转换时间戳
            logger.info("Converting timestamps...")
            df['click_timestamp'] = pd.to_datetime(df['click_timestamp'], unit='s', errors='coerce')
            
            # 处理缺失值（使用默认值）
            logger.info("Handling missing values...")
            df = df.fillna({
                'click_environment': 'unknown',
                'click_deviceGroup': 'unknown',
                'click_os': 'unknown',
                'click_country': 'unknown',
                'click_region': 'unknown',
                'click_referrer_type': 'unknown'
            })
            
            logger.info(f"Loaded {len(df)} training records for {df['user_id'].nunique()} users")
            return df
            
        except FileNotFoundError:
            raise
        except Exception as e:
            raise DataLoadError(f"Error loading training data: {str(e)}")
    
    def split_train_validation(self, click_log: pd.DataFrame) -> tuple:
        """
        将完整数据分割为训练集和验证集
        对于每个用户，最后一条点击作为验证数据，之前的数据作为训练数据
        
        Args:
            click_log: 完整的用户点击日志
            
        Returns:
            (train_data, validation_data) 元组
        """
        try:
            logger.info("Splitting data into train and validation sets...")
            logger.info("  Strategy: Last click per user → validation, previous clicks → training")
            
            # 确保时间戳已转换
            if not pd.api.types.is_datetime64_any_dtype(click_log['click_timestamp']):
                click_log['click_timestamp'] = pd.to_datetime(click_log['click_timestamp'], unit='s', errors='coerce')
            
            # 按用户分组，获取每个用户的最后一条点击
            train_records = []
            validation_records = []
            
            user_groups = click_log.groupby('user_id')
            total_users = len(user_groups)
            
            for idx, (user_id, group) in enumerate(user_groups):
                if (idx + 1) % 10000 == 0:
                    logger.info(f"  Processing users: {idx + 1}/{total_users} ({100*(idx+1)/total_users:.1f}%)")
                
                # 按时间戳排序
                sorted_group = group.sort_values('click_timestamp').copy()
                
                if len(sorted_group) > 1:
                    # 最后一条作为验证数据
                    validation_records.append(sorted_group.iloc[-1:])
                    # 之前的作为训练数据
                    train_records.append(sorted_group.iloc[:-1])
                else:
                    # 如果用户只有一条记录，放入训练集（无法验证）
                    train_records.append(sorted_group)
            
            # 合并训练数据
            if train_records:
                train_data = pd.concat(train_records, ignore_index=True)
            else:
                train_data = pd.DataFrame(columns=click_log.columns)
            
            # 合并验证数据
            if validation_records:
                validation_data = pd.concat(validation_records, ignore_index=True)
            else:
                validation_data = pd.DataFrame(columns=click_log.columns)
            
            logger.info(f"Split completed:")
            logger.info(f"  Training data: {len(train_data)} records for {train_data['user_id'].nunique()} users")
            logger.info(f"  Validation data: {len(validation_data)} records for {validation_data['user_id'].nunique()} users")
            
            return train_data, validation_data
            
        except Exception as e:
            raise DataLoadError(f"Error splitting train/validation data: {str(e)}")
    
    def load_split_data(self, train_path: str = None, validation_path: str = None) -> tuple:
        """
        加载已保存的分割后的训练集和验证集
        
        Args:
            train_path: 训练数据文件路径（默认：data/train_split.csv）
            validation_path: 验证数据文件路径（默认：data/validation_split.csv）
            
        Returns:
            (train_data, validation_data) 元组，如果文件不存在返回 (None, None)
        """
        import os
        from ..utils.config import DATA_DIR
        
        if train_path is None:
            train_path = os.path.join(DATA_DIR, 'train_split.csv')
        if validation_path is None:
            validation_path = os.path.join(DATA_DIR, 'validation_split.csv')
        
        # 检查文件是否存在
        if not os.path.exists(train_path) or not os.path.exists(validation_path):
            return None, None
        
        try:
            logger.info(f"Loading split data from files...")
            logger.info(f"  Training data: {train_path}")
            logger.info(f"  Validation data: {validation_path}")
            
            # 加载训练数据
            train_data = pd.read_csv(train_path)
            # 转换时间戳
            train_data['click_timestamp'] = pd.to_datetime(train_data['click_timestamp'], unit='s', errors='coerce')
            
            # 加载验证数据
            validation_data = pd.read_csv(validation_path)
            # 转换时间戳
            validation_data['click_timestamp'] = pd.to_datetime(validation_data['click_timestamp'], unit='s', errors='coerce')
            
            logger.info(f"Split data loaded successfully")
            logger.info(f"  Training data: {len(train_data)} records for {train_data['user_id'].nunique()} users")
            logger.info(f"  Validation data: {len(validation_data)} records for {validation_data['user_id'].nunique()} users")
            
            return train_data, validation_data
            
        except Exception as e:
            logger.warning(f"Failed to load split data: {str(e)}")
            return None, None
    
    def save_split_data(self, train_data: pd.DataFrame, validation_data: pd.DataFrame, 
                       train_path: str = None, validation_path: str = None) -> None:
        """
        保存分割后的训练集和验证集到文件
        
        Args:
            train_data: 训练数据
            validation_data: 验证数据
            train_path: 训练数据保存路径（默认：data/train_split.csv）
            validation_path: 验证数据保存路径（默认：data/validation_split.csv）
        """
        import os
        from ..utils.config import DATA_DIR
        
        if train_path is None:
            train_path = os.path.join(DATA_DIR, 'train_split.csv')
        if validation_path is None:
            validation_path = os.path.join(DATA_DIR, 'validation_split.csv')
        
        try:
            logger.info(f"Saving split data to files...")
            logger.info(f"  Training data: {train_path}")
            logger.info(f"  Validation data: {validation_path}")
            
            # 保存训练数据（需要将时间戳转换回Unix时间戳）
            train_data_save = train_data.copy()
            if pd.api.types.is_datetime64_any_dtype(train_data_save['click_timestamp']):
                train_data_save['click_timestamp'] = train_data_save['click_timestamp'].astype('int64') // 10**9
            
            # 保存验证数据（需要将时间戳转换回Unix时间戳）
            validation_data_save = validation_data.copy()
            if pd.api.types.is_datetime64_any_dtype(validation_data_save['click_timestamp']):
                validation_data_save['click_timestamp'] = validation_data_save['click_timestamp'].astype('int64') // 10**9
            
            train_data_save.to_csv(train_path, index=False)
            validation_data_save.to_csv(validation_path, index=False)
            
            logger.info(f"Split data saved successfully")
            logger.info(f"  Training data: {len(train_data)} records → {train_path}")
            logger.info(f"  Validation data: {len(validation_data)} records → {validation_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save split data: {str(e)}")
            raise DataLoadError(f"Error saving split data: {str(e)}")
    
    def load_test_data(self, file_path: str) -> pd.DataFrame:
        """
        加载测试集用户点击日志
        
        Args:
            file_path: 测试集CSV文件路径
            
        Returns:
            包含用户点击记录的DataFrame（格式同训练集）
            
        Raises:
            FileNotFoundError: 文件不存在
            DataLoadError: 数据加载错误
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Test data file not found: {file_path}")
            
            logger.info(f"Loading test data from {file_path}")
            logger.info("Reading CSV file...")
            df = pd.read_csv(file_path)
            logger.info(f"CSV file loaded: {len(df)} rows")
            
            # 验证必需的列
            logger.info("Validating data columns...")
            validate_dataframe_columns(df, self.CLICK_LOG_COLUMNS)
            
            # 数据验证
            if df.empty:
                raise DataLoadError("Test data is empty")
            
            # 转换时间戳
            logger.info("Converting timestamps...")
            df['click_timestamp'] = pd.to_datetime(df['click_timestamp'], unit='s', errors='coerce')
            
            # 处理缺失值（使用默认值）
            logger.info("Handling missing values...")
            df = df.fillna({
                'click_environment': 'unknown',
                'click_deviceGroup': 'unknown',
                'click_os': 'unknown',
                'click_country': 'unknown',
                'click_region': 'unknown',
                'click_referrer_type': 'unknown'
            })
            
            logger.info(f"Loaded {len(df)} test records for {df['user_id'].nunique()} users")
            return df
            
        except FileNotFoundError:
            raise
        except Exception as e:
            raise DataLoadError(f"Error loading test data: {str(e)}")
    
    def load_articles(self, file_path: str) -> pd.DataFrame:
        """
        加载文章信息数据
        
        Args:
            file_path: 文章信息CSV文件路径
            
        Returns:
            包含文章信息的DataFrame
            
        Raises:
            FileNotFoundError: 文件不存在
            DataLoadError: 数据加载错误
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Articles file not found: {file_path}")
            
            logger.info(f"Loading articles from {file_path}")
            logger.info("Reading CSV file...")
            df = pd.read_csv(file_path)
            logger.info(f"CSV file loaded: {len(df)} rows")
            
            # 验证必需的列
            logger.info("Validating data columns...")
            validate_dataframe_columns(df, self.ARTICLES_COLUMNS)
            
            # 数据验证
            if df.empty:
                raise DataLoadError("Articles data is empty")
            
            # 处理缺失值
            logger.info("Handling missing values...")
            df = df.fillna({
                'category_id': 0,
                'created_at_ts': 0,
                'words_count': 0
            })
            
            logger.info(f"Loaded {len(df)} articles")
            return df
            
        except FileNotFoundError:
            raise
        except Exception as e:
            raise DataLoadError(f"Error loading articles: {str(e)}")
    
    def load_article_embeddings(self, file_path: str) -> np.ndarray:
        """
        加载文章embedding向量
        
        Args:
            file_path: 文章embedding CSV文件路径
            
        Returns:
            文章embedding向量矩阵，形状为(n_articles, dim)，维度自动检测
            
        Raises:
            FileNotFoundError: 文件不存在
            DataLoadError: 数据加载错误
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Article embeddings file not found: {file_path}")
            
            logger.info(f"Loading article embeddings from {file_path}")
            logger.info("Reading large CSV file in chunks (this may take a while)...")
            
            # 分块读取大文件（内存优化）
            chunk_size = 10000
            chunks = []
            article_ids = []
            total_chunks = 0
            
            # 先读取一行以确定总行数（估算）
            first_chunk = pd.read_csv(file_path, nrows=1)
            if 'article_id' in first_chunk.columns:
                has_article_id = True
            else:
                has_article_id = False
            
            # 重新读取文件（分块）
            for chunk_idx, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
                total_chunks += 1
                if total_chunks % 10 == 0:
                    logger.info(f"  Processed {total_chunks * chunk_size} rows...")
                
                # 提取article_id
                if has_article_id:
                    article_ids.extend(chunk['article_id'].tolist())
                    # 提取embedding列（emb_0到emb_249）
                    emb_columns = [col for col in chunk.columns if col.startswith('emb_')]
                    emb_data = chunk[emb_columns].values
                else:
                    # 如果没有article_id列，假设第一列是article_id
                    article_ids.extend(chunk.iloc[:, 0].tolist())
                    emb_columns = [col for col in chunk.columns[1:] if col.startswith('emb_')]
                    emb_data = chunk[emb_columns].values
                
                chunks.append(emb_data)
            
            # 合并所有chunks
            if chunks:
                embeddings = np.vstack(chunks)
            else:
                raise DataLoadError("No embedding data found")
            
            # 自动检测并记录维度
            actual_dim = embeddings.shape[1]
            if self.EMBEDDING_DIM is None:
                self.EMBEDDING_DIM = actual_dim
                logger.info(f"Detected embedding dimension: {actual_dim}")
            elif embeddings.shape[1] != self.EMBEDDING_DIM:
                logger.warning(
                    f"Embedding dimension mismatch: expected {self.EMBEDDING_DIM}, "
                    f"got {actual_dim}. Using actual dimension {actual_dim}."
                )
                self.EMBEDDING_DIM = actual_dim
            
            # 处理缺失值（使用零向量填充）
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
            
            logger.info(f"Loaded {len(embeddings)} article embeddings with dimension {embeddings.shape[1]}")
            return embeddings
            
        except FileNotFoundError:
            raise
        except Exception as e:
            raise DataLoadError(f"Error loading article embeddings: {str(e)}")

