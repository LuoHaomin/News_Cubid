"""
异常类定义
"""


class BaselineRecommenderError(Exception):
    """基线推荐系统基础异常类"""
    pass


class DataLoadError(BaselineRecommenderError):
    """数据加载错误"""
    pass


class DataValidationError(BaselineRecommenderError):
    """数据验证错误"""
    pass


class ModelTrainingError(BaselineRecommenderError):
    """模型训练错误"""
    pass


class RecommendationError(BaselineRecommenderError):
    """推荐生成错误"""
    pass


class EvaluationError(BaselineRecommenderError):
    """评估计算错误"""
    pass


class OutputError(BaselineRecommenderError):
    """结果输出错误"""
    pass

