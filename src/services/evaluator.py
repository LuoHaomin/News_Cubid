"""
评估服务模块
"""

import pandas as pd
from typing import Dict, List

from ..utils.metrics import calculate_mrr
from ..utils.exceptions import EvaluationError
from ..utils.logger import logger


class Evaluator:
    """评估器类"""
    
    def extract_ground_truth(self, test_click_log: pd.DataFrame) -> Dict[str, str]:
        """
        提取真实标签（每个用户的最后一次点击）
        
        Args:
            test_click_log: 测试集点击日志
            
        Returns:
            真实标签字典，key为用户ID，value为真实点击文章ID
        """
        logger.info("Extracting ground truth (last click per user)...")
        
        # 按用户分组，获取每个用户的最后一次点击
        ground_truth = {}
        user_groups = test_click_log.groupby('user_id')
        total_users = len(user_groups)
        
        for idx, (user_id, group) in enumerate(user_groups):
            if (idx + 1) % 5000 == 0:
                logger.info(f"  Processing users: {idx + 1}/{total_users} ({100*(idx+1)/total_users:.1f}%)")
            
            # 按时间戳排序，取最后一次点击
            sorted_group = group.sort_values('click_timestamp')
            last_click = sorted_group.iloc[-1]
            ground_truth[user_id] = last_click['click_article_id']
        
        logger.info(f"Extracted ground truth for {len(ground_truth)} users")
        return ground_truth
    
    def calculate_mrr(self, recommendations: Dict[str, List[str]], ground_truth: Dict[str, str]) -> float:
        """
        计算MRR指标
        
        Args:
            recommendations: 推荐结果字典
            ground_truth: 真实标签字典
            
        Returns:
            MRR值
        """
        return calculate_mrr(recommendations, ground_truth)
    
    def evaluate(self, recommendations: Dict[str, List[str]], test_click_log: pd.DataFrame) -> Dict:
        """
        评估推荐效果
        
        Args:
            recommendations: 推荐结果字典
            test_click_log: 测试集点击日志
            
        Returns:
            评估结果字典，包含MRR值和统计信息
        """
        try:
            logger.info("Evaluating recommendations...")
            
            # 提取真实标签
            ground_truth = self.extract_ground_truth(test_click_log)
            
            # 计算MRR
            mrr = self.calculate_mrr(recommendations, ground_truth)
            
            # 统计信息
            total_users = len(recommendations)
            total_recommendations = sum(len(recs) for recs in recommendations.values())
            
            result = {
                'mrr': mrr,
                'total_users': total_users,
                'total_recommendations': total_recommendations,
                'avg_recommendations_per_user': total_recommendations / total_users if total_users > 0 else 0
            }
            
            logger.info(f"Evaluation completed: MRR = {mrr:.4f}, Total users = {total_users}")
            return result
            
        except Exception as e:
            raise EvaluationError(f"Error evaluating recommendations: {str(e)}")

