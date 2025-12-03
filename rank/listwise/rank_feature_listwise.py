import argparse
import os
import subprocess
import sys
import warnings

import pandas as pd

from common.utils import Logger

# 获取项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

warnings.filterwarnings('ignore')

# 命令行参数
parser = argparse.ArgumentParser(description='Listwise 排序特征（在 Pointwise 特征基础上追加列表特征）')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
log_dir = os.path.join(root_dir, 'user_data', 'log')
os.makedirs(log_dir, exist_ok=True)
log = Logger(os.path.join(log_dir, logfile)).logger
log.info(f'Listwise 排序特征，mode: {mode}')

DATA_DIR = os.path.join(root_dir, 'user_data', 'data', 'offline' if mode == 'valid' else 'online')
POINTWISE_FEATURE_FILE = os.path.join(DATA_DIR, 'feature.pkl')
LISTWISE_FEATURE_FILE = os.path.join(DATA_DIR, 'feature_listwise.pkl')

LIST_FEATURE_COLUMNS = [
    'list_size',
    'list_rank',
    'list_category_diversity',
    'list_category_coverage',
    'list_time_span',
    'list_avg_words_count',
    'list_std_words_count',
    'list_avg_sim_score',
    'list_std_sim_score',
    'list_max_sim_score',
    'list_min_sim_score',
    'words_count_diff_from_avg',
    'sim_score_diff_from_avg',
    'sim_score_diff_from_max',
]


def ensure_pointwise_feature():
    """确认 pointwise 特征文件已存在，如不存在则自动调用 pointwise 脚本生成。"""
    if os.path.exists(POINTWISE_FEATURE_FILE):
        log.info(f'检测到 pointwise 特征文件: {POINTWISE_FEATURE_FILE}')
        return

    log.warning('未检测到 pointwise 特征文件，开始调用 pointwise 特征脚本...')
    pointwise_script = os.path.join(root_dir, 'rank', 'pointwise', 'rank_feature.py')
    cmd = [
        sys.executable,
        pointwise_script,
        '--mode',
        mode,
        '--logfile',
        f'auto_pointwise_{mode}.log',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.error('pointwise 特征脚本执行失败')
        log.error(result.stderr.strip())
        raise RuntimeError('pointwise 特征脚本执行失败，请先生成 feature.pkl')

    if not os.path.exists(POINTWISE_FEATURE_FILE):
        raise FileNotFoundError('pointwise 特征脚本执行完成，但未找到 feature.pkl')

    log.info('pointwise 特征生成完成')


def add_listwise_features(df_feature: pd.DataFrame) -> pd.DataFrame:
    """在已有特征的基础上，追加列表级别统计特征。"""
    required_columns = [
        'user_id',
        'article_id',
        'category_id',
        'created_at_ts',
        'words_count',
    ]
    missing_cols = [col for col in required_columns if col not in df_feature.columns]
    if missing_cols:
        raise KeyError(f'缺少计算列表特征所需字段: {missing_cols}')

    df = df_feature.copy()

    # 为避免 group 长度与数据不一致，确保 user_id 顺序稳定
    sort_cols = ['user_id']
    sort_orders = [True]
    if 'sim_score' in df.columns:
        sort_cols.append('sim_score')
        sort_orders.append(False)
    df.sort_values(sort_cols, ascending=sort_orders, inplace=True, na_position='last')
    df.reset_index(drop=True, inplace=True)

    # 列表规模与多样性
    df['list_size'] = df.groupby('user_id')['article_id'].transform('count')
    df['list_category_diversity'] = df.groupby('user_id')['category_id'].transform('nunique')
    df['list_category_coverage'] = df['list_category_diversity'] / df['list_size']

    # 时间跨度
    df['list_time_span'] = df.groupby('user_id')['created_at_ts'].transform(lambda x: x.max() - x.min())
    df['list_time_span'] = df['list_time_span'].fillna(0)

    # 字数统计
    df['list_avg_words_count'] = df.groupby('user_id')['words_count'].transform('mean')
    df['list_std_words_count'] = df.groupby('user_id')['words_count'].transform('std').fillna(0)
    df['words_count_diff_from_avg'] = df['words_count'] - df['list_avg_words_count']

    # rank
    if 'sim_score' in df.columns:
        df['list_rank'] = df.groupby('user_id')['sim_score'].rank(method='first', ascending=False)
        df['list_avg_sim_score'] = df.groupby('user_id')['sim_score'].transform('mean')
        df['list_std_sim_score'] = df.groupby('user_id')['sim_score'].transform('std').fillna(0)
        df['list_max_sim_score'] = df.groupby('user_id')['sim_score'].transform('max')
        df['list_min_sim_score'] = df.groupby('user_id')['sim_score'].transform('min')
        df['sim_score_diff_from_avg'] = df['sim_score'] - df['list_avg_sim_score']
        df['sim_score_diff_from_max'] = df['sim_score'] - df['list_max_sim_score']
    else:
        # 兼容缺少 sim_score 的场景
        df['list_rank'] = 1
        df['list_avg_sim_score'] = 0
        df['list_std_sim_score'] = 0
        df['list_max_sim_score'] = 0
        df['list_min_sim_score'] = 0
        df['sim_score_diff_from_avg'] = 0
        df['sim_score_diff_from_max'] = 0

    log.info(f'列表级别特征添加完成，新增列数: {len(LIST_FEATURE_COLUMNS)}')
    return df


def main():
    ensure_pointwise_feature()

    df_feature = pd.read_pickle(POINTWISE_FEATURE_FILE)
    log.info(f'载入 pointwise 特征: {df_feature.shape}')

    # 如果重复运行，需要先移除旧的列表特征
    duplicated_cols = [col for col in LIST_FEATURE_COLUMNS if col in df_feature.columns]
    if duplicated_cols:
        log.warning(f'检测到已存在的列表特征列，将先删除: {duplicated_cols}')
        df_feature = df_feature.drop(columns=duplicated_cols)

    df_feature = add_listwise_features(df_feature)
    df_feature.to_pickle(LISTWISE_FEATURE_FILE)
    log.info(f'Listwise 特征已保存到: {LISTWISE_FEATURE_FILE}')


if __name__ == '__main__':
    main()

