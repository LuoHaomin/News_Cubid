"""
用户模型
"""

from typing import List, Optional
from .click_log import ClickLog


class User:
    """用户实体类"""
    
    def __init__(self, user_id: str, click_history: Optional[List[ClickLog]] = None):
        """
        初始化用户
        
        Args:
            user_id: 用户唯一标识符
            click_history: 用户历史点击记录列表
        """
        self.user_id = user_id
        self.click_history = click_history or []
        self.click_count = len(self.click_history)
    
    def add_click_log(self, click_log: ClickLog) -> None:
        """
        添加点击记录
        
        Args:
            click_log: 点击记录对象
        """
        self.click_history.append(click_log)
        self.click_count = len(self.click_history)
    
    def get_clicked_article_ids(self) -> List[str]:
        """
        获取用户点击过的文章ID列表
        
        Returns:
            文章ID列表
        """
        return [click_log.click_article_id for click_log in self.click_history]
    
    def __repr__(self) -> str:
        return f"User(user_id={self.user_id}, click_count={self.click_count})"

