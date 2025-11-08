"""
数据模型模块
"""

from .user import User
from .article import Article
from .click_log import ClickLog
from .recommendation import Recommendation

__all__ = ['User', 'Article', 'ClickLog', 'Recommendation']
