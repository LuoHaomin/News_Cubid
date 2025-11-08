"""
点击记录模型
"""

from typing import Optional


class ClickLog:
    """点击记录实体类"""
    
    def __init__(self,
                 user_id: str,
                 click_article_id: str,
                 click_timestamp: int,
                 click_environment: Optional[str] = None,
                 click_deviceGroup: Optional[str] = None,
                 click_os: Optional[str] = None,
                 click_country: Optional[str] = None,
                 click_region: Optional[str] = None,
                 click_referrer_type: Optional[str] = None):
        """
        初始化点击记录
        
        Args:
            user_id: 用户ID
            click_article_id: 点击的文章ID
            click_timestamp: 点击时间戳
            click_environment: 点击环境
            click_deviceGroup: 点击设备组
            click_os: 点击操作系统
            click_country: 点击城市
            click_region: 点击地区
            click_referrer_type: 点击来源类型
        """
        self.user_id = user_id
        self.click_article_id = click_article_id
        self.click_timestamp = click_timestamp
        self.click_environment = click_environment
        self.click_deviceGroup = click_deviceGroup
        self.click_os = click_os
        self.click_country = click_country
        self.click_region = click_region
        self.click_referrer_type = click_referrer_type
    
    def __repr__(self) -> str:
        return f"ClickLog(user_id={self.user_id}, article_id={self.click_article_id}, timestamp={self.click_timestamp})"

