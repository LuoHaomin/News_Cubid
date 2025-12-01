import os

def get_root_dir():
    """获取项目根目录"""
    current_file = os.path.abspath(__file__)
    # common/paths.py -> 项目根目录
    return os.path.dirname(os.path.dirname(current_file))

def get_data_dir():
    """获取数据目录"""
    return os.path.join(get_root_dir(), 'data')

def get_user_data_dir():
    """获取用户数据目录"""
    return os.path.join(get_root_dir(), 'user_data')

def get_user_data_path(*paths):
    """获取用户数据目录下的路径"""
    return os.path.join(get_user_data_dir(), *paths)

def get_data_path(*paths):
    """获取数据目录下的路径"""
    return os.path.join(get_data_dir(), *paths)

