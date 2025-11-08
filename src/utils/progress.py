"""
进度显示工具模块
"""

import sys
import time
from typing import Optional


class ProgressBar:
    """简单的进度条实现"""
    
    def __init__(self, total: int, desc: str = "Progress", width: int = 50):
        """
        初始化进度条
        
        Args:
            total: 总任务数
            desc: 描述文本
            width: 进度条宽度
        """
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n: int = 1):
        """
        更新进度
        
        Args:
            n: 增加的任务数
        """
        self.current += n
        self._display()
    
    def _display(self):
        """显示进度条"""
        if self.total == 0:
            percent = 0
        else:
            percent = min(100, 100 * self.current / self.total)
        
        filled = int(self.width * self.current / self.total) if self.total > 0 else 0
        bar = '=' * filled + '-' * (self.width - filled)
        
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
        else:
            eta = 0
        
        sys.stdout.write(f'\r{self.desc}: [{bar}] {self.current}/{self.total} ({percent:.1f}%) | '
                        f'Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s')
        sys.stdout.flush()
    
    def close(self):
        """关闭进度条"""
        sys.stdout.write('\n')
        sys.stdout.flush()


def show_progress(iterable, desc: str = "Processing", total: Optional[int] = None):
    """
    显示迭代进度（简化版tqdm）
    
    Args:
        iterable: 可迭代对象
        desc: 描述文本
        total: 总数量（如果iterable没有__len__方法）
        
    Yields:
        迭代项
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None
    
    if total is None:
        # 如果没有总数，只显示当前项
        for i, item in enumerate(iterable):
            sys.stdout.write(f'\r{desc}: {i+1} items processed')
            sys.stdout.flush()
            yield item
        sys.stdout.write('\n')
    else:
        # 有总数，显示进度条
        bar = ProgressBar(total, desc)
        for item in iterable:
            yield item
            bar.update(1)
        bar.close()

