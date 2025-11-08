"""
结果输出服务模块
"""

import pandas as pd
from typing import Dict, List
import os

from ..utils.exceptions import OutputError
from ..utils.validation import validate_recommendation_list
from ..utils.logger import logger
from ..utils.config import TOP_K


class OutputWriter:
    """结果输出器类"""
    
    def write_submission(self, recommendations: Dict[str, List[str]], output_path: str) -> None:
        """
        将推荐结果写入提交文件
        
        Args:
            recommendations: 推荐结果字典，key为用户ID，value为推荐文章ID列表
            output_path: 输出文件路径
            
        Raises:
            OutputError: 输出错误
        """
        try:
            logger.info(f"Writing submission file to {output_path}")
            
            logger.info(f"Validating {len(recommendations)} recommendations...")
            # 验证推荐结果格式
            invalid_count = 0
            for user_id, article_ids in recommendations.items():
                try:
                    validate_recommendation_list(article_ids, TOP_K)
                except Exception as e:
                    logger.warning(f"Invalid recommendation format for user {user_id}: {str(e)}")
                    invalid_count += 1
                    # 修复格式：确保有5篇文章
                    if len(article_ids) < TOP_K:
                        # 用空字符串填充
                        article_ids.extend([''] * (TOP_K - len(article_ids)))
            
            if invalid_count > 0:
                logger.warning(f"Found {invalid_count} invalid recommendations, fixed automatically")
            
            # 准备数据
            logger.info("Preparing submission data...")
            data = []
            for user_id, article_ids in recommendations.items():
                row = {
                    'user_id': user_id,
                    'article_1': article_ids[0] if len(article_ids) > 0 else '',
                    'article_2': article_ids[1] if len(article_ids) > 1 else '',
                    'article_3': article_ids[2] if len(article_ids) > 2 else '',
                    'article_4': article_ids[3] if len(article_ids) > 3 else '',
                    'article_5': article_ids[4] if len(article_ids) > 4 else ''
                }
                data.append(row)
            
            # 创建DataFrame
            df = pd.DataFrame(data)
            
            # 确保列顺序正确
            columns = ['user_id', 'article_1', 'article_2', 'article_3', 'article_4', 'article_5']
            df = df[columns]
            
            # 创建输出目录（如果不存在）
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 写入CSV文件
            df.to_csv(output_path, index=False)
            
            logger.info(f"Submission file written successfully: {len(df)} users")
            
        except Exception as e:
            raise OutputError(f"Error writing submission file: {str(e)}")

