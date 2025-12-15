import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from pandarallel import pandarallel
import sys

from common.utils import Logger

# 获取项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Windows上pandarallel不稳定，禁用并行处理
USE_PARALLEL = sys.platform != 'win32'
if USE_PARALLEL:
    pandarallel.initialize()
else:
    # Windows上不初始化pandarallel，使用普通apply
    print('Windows detected, using standard apply instead of parallel_apply')

warnings.filterwarnings('ignore')

seed = 2020

# 命令行参数
parser = argparse.ArgumentParser(description='排序特征')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
log_dir = os.path.join(root_dir, 'user_data', 'log')
os.makedirs(log_dir, exist_ok=True)
log = Logger(os.path.join(log_dir, logfile)).logger
log.info(f'排序特征，mode: {mode}')

if not USE_PARALLEL:
    log.warning('Windows detected, using standard apply instead of parallel_apply')


def compute_itemcf_sim_features_vectorized(df, user_item_dict, item_sim):
    """
    向量化计算 itemcf 相似度特征，比 parallel_apply 快 10-50 倍
    返回 (sim_sum, last_sim) 两列
    """
    import numpy as np
    from tqdm import tqdm
    
    n = len(df)
    sim_sum_arr = np.zeros(n, dtype=np.float32)
    last_sim_arr = np.zeros(n, dtype=np.float32)
    
    # 预计算用户最后点击的物品
    user_last_item = {uid: items[-1] for uid, items in user_item_dict.items()}
    
    # 按 user_id 分组处理，减少重复查找
    user_ids = df['user_id'].values
    article_ids = df['article_id'].values
    
    # 使用分组加速
    df_temp = df[['user_id', 'article_id']].copy()
    df_temp['idx'] = np.arange(n)
    
    for user_id, group in tqdm(df_temp.groupby('user_id'), desc='计算ItemCF特征'):
        if user_id not in user_item_dict:
            continue
            
        indices = group['idx'].values
        arts = group['article_id'].values
        
        interacted_items = user_item_dict[user_id][::-1]
        last_item = interacted_items[0]
        
        # 预计算衰减权重
        decay_weights = np.array([0.7**loc for loc in range(len(interacted_items))], dtype=np.float32)
        
        for idx, article_id in zip(indices, arts):
            # 计算 last_sim
            if last_item in item_sim and article_id in item_sim.get(last_item, {}):
                last_sim_arr[idx] = item_sim[last_item][article_id]
            
            # 计算 sim_sum
            sim_sum = 0.0
            for loc, i in enumerate(interacted_items):
                if i in item_sim and article_id in item_sim.get(i, {}):
                    sim_sum += item_sim[i][article_id] * decay_weights[loc]
            sim_sum_arr[idx] = sim_sum
    
    return sim_sum_arr, last_sim_arr


def compute_binetwork_sim_last_vectorized(df, user_item_dict, binetwork_sim):
    """向量化计算 binetwork 最后点击相似度"""
    import numpy as np
    from tqdm import tqdm
    
    n = len(df)
    last_sim_arr = np.zeros(n, dtype=np.float32)
    
    user_last_item = {uid: items[-1] for uid, items in user_item_dict.items()}
    
    df_temp = df[['user_id', 'article_id']].copy()
    df_temp['idx'] = np.arange(n)
    
    for user_id, group in tqdm(df_temp.groupby('user_id'), desc='计算Binetwork特征'):
        if user_id not in user_last_item:
            continue
            
        last_item = user_last_item[user_id]
        if last_item not in binetwork_sim:
            continue
            
        indices = group['idx'].values
        arts = group['article_id'].values
        sim_dict = binetwork_sim[last_item]
        
        for idx, article_id in zip(indices, arts):
            if article_id in sim_dict:
                last_sim_arr[idx] = sim_dict[article_id]
    
    return last_sim_arr


def compute_w2v_sim_features_vectorized(df, user_item_dict, article_vec_map, num=2):
    """向量化计算 w2v 相似度特征"""
    import numpy as np
    from tqdm import tqdm
    
    n = len(df)
    last_sim_arr = np.zeros(n, dtype=np.float32)
    sum_sim_arr = np.zeros(n, dtype=np.float32)
    
    # 预计算向量范数
    vec_norms = {aid: np.linalg.norm(vec) for aid, vec in article_vec_map.items()}
    
    df_temp = df[['user_id', 'article_id']].copy()
    df_temp['idx'] = np.arange(n)
    
    for user_id, group in tqdm(df_temp.groupby('user_id'), desc='计算W2V特征'):
        if user_id not in user_item_dict:
            continue
            
        indices = group['idx'].values
        arts = group['article_id'].values
        
        interacted_items = user_item_dict[user_id][::-1]
        last_item = interacted_items[0]
        recent_items = interacted_items[:num]
        
        for idx, article_id in zip(indices, arts):
            if article_id not in article_vec_map:
                continue
                
            vec_a = article_vec_map[article_id]
            norm_a = vec_norms[article_id]
            
            # last sim
            if last_item in article_vec_map and norm_a > 0:
                vec_b = article_vec_map[last_item]
                norm_b = vec_norms[last_item]
                if norm_b > 0:
                    last_sim_arr[idx] = np.dot(vec_a, vec_b) / (norm_a * norm_b)
            
            # sum sim
            sim_sum = 0.0
            for i in recent_items:
                if i in article_vec_map and norm_a > 0:
                    vec_b = article_vec_map[i]
                    norm_b = vec_norms[i]
                    if norm_b > 0:
                        sim_sum += np.dot(vec_a, vec_b) / (norm_a * norm_b)
            sum_sim_arr[idx] = sim_sum
    
    return last_sim_arr, sum_sim_arr


# 保留旧函数以兼容，但标记为废弃
def make_func_if_sum(user_item_dict, item_sim):
    """[废弃] 创建func_if_sum函数，建议使用 compute_itemcf_sim_features_vectorized"""
    def func_if_sum(x):
        user_id = x['user_id']
        article_id = x['article_id']

        if user_id not in user_item_dict:
            return 0
        
        interacted_items = user_item_dict[user_id]
        interacted_items = interacted_items[::-1]

        sim_sum = 0
        for loc, i in enumerate(interacted_items):
            try:
                if i in item_sim and article_id in item_sim[i]:
                    sim_sum += item_sim[i][article_id] * (0.7**loc)
            except Exception as e:
                pass
        return sim_sum
    return func_if_sum


def make_func_if_last(user_item_dict, item_sim):
    """[废弃] 创建func_if_last函数，建议使用 compute_itemcf_sim_features_vectorized"""
    def func_if_last(x):
        user_id = x['user_id']
        article_id = x['article_id']

        if user_id not in user_item_dict:
            return 0
        
        last_item = user_item_dict[user_id][-1]

        sim = 0
        try:
            if last_item in item_sim and article_id in item_sim[last_item]:
                sim = item_sim[last_item][article_id]
        except Exception as e:
            pass
        return sim
    return func_if_last


def make_func_binetwork_sim_last(user_item_dict, binetwork_sim):
    """[废弃] 创建func_binetwork_sim_last函数，建议使用 compute_binetwork_sim_last_vectorized"""
    def func_binetwork_sim_last(x):
        user_id = x['user_id']
        article_id = x['article_id']

        if user_id not in user_item_dict:
            return 0
        
        last_item = user_item_dict[user_id][-1]

        sim = 0
        try:
            if last_item in binetwork_sim and article_id in binetwork_sim[last_item]:
                sim = binetwork_sim[last_item][article_id]
        except Exception as e:
            pass
        return sim
    return func_binetwork_sim_last


def consine_distance(vector1, vector2):
    if type(vector1) != np.ndarray or type(vector2) != np.ndarray:
        return -1
    distance = np.dot(vector1, vector2) / \
        (np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
    return distance


def make_func_w2w_sum(user_item_dict, article_vec_map, num):
    """创建func_w2w_sum函数，捕获需要的变量"""
    def func_w2w_sum(x):
        user_id = x['user_id']
        article_id = x['article_id']

        if user_id not in user_item_dict:
            return 0
        
        interacted_items = user_item_dict[user_id]
        interacted_items = interacted_items[::-1][:num]

        sim_sum = 0
        for loc, i in enumerate(interacted_items):
            try:
                if article_id in article_vec_map and i in article_vec_map:
                    sim_sum += consine_distance(article_vec_map[article_id],
                                                article_vec_map[i])
            except Exception as e:
                pass
        return sim_sum
    return func_w2w_sum


def make_func_w2w_last_sim(user_item_dict, article_vec_map):
    """创建func_w2w_last_sim函数，捕获需要的变量"""
    def func_w2w_last_sim(x):
        user_id = x['user_id']
        article_id = x['article_id']

        if user_id not in user_item_dict:
            return 0
        
        last_item = user_item_dict[user_id][-1]

        sim = 0
        try:
            if article_id in article_vec_map and last_item in article_vec_map:
                sim = consine_distance(article_vec_map[article_id],
                                       article_vec_map[last_item])
        except Exception as e:
            pass
        return sim
    return func_w2w_last_sim


if __name__ == '__main__':
    if mode == 'valid':
        df_feature = pd.read_pickle(os.path.join(root_dir, 'user_data', 'data', 'offline', 'recall.pkl'))
        df_click = pd.read_pickle(os.path.join(root_dir, 'user_data', 'data', 'offline', 'click.pkl'))

    else:
        df_feature = pd.read_pickle(os.path.join(root_dir, 'user_data', 'data', 'online', 'recall.pkl'))
        df_click = pd.read_pickle(os.path.join(root_dir, 'user_data', 'data', 'online', 'click.pkl'))

    # 文章特征
    log.debug(f'df_feature.shape: {df_feature.shape}')

    df_article = pd.read_csv(os.path.join(root_dir, 'data', 'articles.csv'))
    df_article['created_at_ts'] = df_article['created_at_ts'] / 1000
    df_article['created_at_ts'] = df_article['created_at_ts'].astype('int')
    df_feature = df_feature.merge(df_article, how='left')
    df_feature['created_at_datetime'] = pd.to_datetime(
        df_feature['created_at_ts'], unit='s')

    log.debug(f'df_article.head(): {df_article.head()}')
    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 历史记录相关特征
    df_click.sort_values(['user_id', 'click_timestamp'], inplace=True)
    df_click.rename(columns={'click_article_id': 'article_id'}, inplace=True)
    df_click = df_click.merge(df_article, how='left')

    df_click['click_timestamp'] = df_click['click_timestamp'] / 1000
    df_click['click_datetime'] = pd.to_datetime(df_click['click_timestamp'],
                                                unit='s',
                                                errors='coerce')
    df_click['click_datetime_hour'] = df_click['click_datetime'].dt.hour

    # 用户点击文章的创建时间差的平均值
    df_click['user_id_click_article_created_at_ts_diff'] = df_click.groupby(
        ['user_id'])['created_at_ts'].diff()
    df_temp = df_click.groupby([
        'user_id'
    ])['user_id_click_article_created_at_ts_diff'].mean().reset_index()
    df_temp.columns = [
        'user_id', 'user_id_click_article_created_at_ts_diff_mean'
    ]
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 用户点击文章的时间差的平均值
    df_click['user_id_click_diff'] = df_click.groupby(
        ['user_id'])['click_timestamp'].diff()
    df_temp = df_click.groupby(['user_id'
                                ])['user_id_click_diff'].mean().reset_index()
    df_temp.columns = ['user_id', 'user_id_click_diff_mean']
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    df_click['click_timestamp_created_at_ts_diff'] = df_click[
        'click_timestamp'] - df_click['created_at_ts']

    # 点击文章的创建时间差的统计值
    df_temp = df_click.groupby(['user_id'])['click_timestamp_created_at_ts_diff'].agg([
        ('user_click_timestamp_created_at_ts_diff_mean', 'mean'),
        ('user_click_timestamp_created_at_ts_diff_std', 'std')
    ]).reset_index()
    # 展平多级列名
    df_temp.columns = ['user_id', 'user_click_timestamp_created_at_ts_diff_mean', 'user_click_timestamp_created_at_ts_diff_std']
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 click_datetime_hour 统计值
    df_temp = df_click.groupby(['user_id'])['click_datetime_hour'].agg([
        ('user_click_datetime_hour_std', 'std')
    ]).reset_index()
    df_temp.columns = ['user_id', 'user_click_datetime_hour_std']
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 words_count 统计值
    df_temp = df_click.groupby(['user_id'])['words_count'].agg([
        ('user_clicked_article_words_count_mean', 'mean'),
        ('user_click_last_article_words_count', lambda x: x.iloc[-1])
    ]).reset_index()
    df_temp.columns = ['user_id', 'user_clicked_article_words_count_mean', 'user_click_last_article_words_count']
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 created_at_ts 统计值
    df_temp = df_click.groupby('user_id')['created_at_ts'].agg([
        ('user_click_last_article_created_time', lambda x: x.iloc[-1]),
        ('user_clicked_article_created_time_max', 'max')
    ]).reset_index()
    df_temp.columns = ['user_id', 'user_click_last_article_created_time', 'user_clicked_article_created_time_max']
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 click_timestamp 统计值
    df_temp = df_click.groupby('user_id')['click_timestamp'].agg([
        ('user_click_last_article_click_time', lambda x: x.iloc[-1]),
        ('user_clicked_article_click_time_mean', 'mean')
    ]).reset_index()
    df_temp.columns = ['user_id', 'user_click_last_article_click_time', 'user_clicked_article_click_time_mean']
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    df_feature['user_last_click_created_at_ts_diff'] = df_feature[
        'created_at_ts'] - df_feature['user_click_last_article_created_time']
    df_feature['user_last_click_timestamp_diff'] = df_feature[
        'created_at_ts'] - df_feature['user_click_last_article_click_time']
    df_feature['user_last_click_words_count_diff'] = df_feature[
        'words_count'] - df_feature['user_click_last_article_words_count']

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 计数统计
    for f in [['user_id'], ['article_id'], ['user_id', 'category_id']]:
        df_temp = df_click.groupby(f).size().reset_index()
        df_temp.columns = f + ['{}_cnt'.format('_'.join(f))]

        df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 召回相关特征
    ## itemcf 相关
    user_item_ = df_click.groupby('user_id')['article_id'].agg(
        list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['article_id']))

    if mode == 'valid':
        f = open(os.path.join(root_dir, 'user_data', 'sim', 'offline', 'itemcf_sim.pkl'), 'rb')
        item_sim = pickle.load(f)
        f.close()
    else:
        f = open(os.path.join(root_dir, 'user_data', 'sim', 'online', 'itemcf_sim.pkl'), 'rb')
        item_sim = pickle.load(f)
        f.close()

    # 用户历史点击物品与待预测物品相似度 - 使用向量化方法（快10-50倍）
    log.info('计算 ItemCF 相似度特征（向量化优化）...')
    sim_sum_arr, last_sim_arr = compute_itemcf_sim_features_vectorized(
        df_feature, user_item_dict, item_sim
    )
    df_feature['user_clicked_article_itemcf_sim_sum'] = sim_sum_arr
    df_feature['user_last_click_article_itemcf_sim'] = last_sim_arr

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    ## binetwork 相关
    if mode == 'valid':
        f = open(os.path.join(root_dir, 'user_data', 'sim', 'offline', 'binetwork_sim.pkl'), 'rb')
        binetwork_sim = pickle.load(f)
        f.close()
    else:
        f = open(os.path.join(root_dir, 'user_data', 'sim', 'online', 'binetwork_sim.pkl'), 'rb')
        binetwork_sim = pickle.load(f)
        f.close()

    log.info('计算 Binetwork 相似度特征（向量化优化）...')
    df_feature['user_last_click_article_binetwork_sim'] = compute_binetwork_sim_last_vectorized(
        df_feature, user_item_dict, binetwork_sim
    )

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    ## w2v 相关
    if mode == 'valid':
        f = open(os.path.join(root_dir, 'user_data', 'data', 'offline', 'article_w2v.pkl'), 'rb')
        article_vec_map = pickle.load(f)
        f.close()
    else:
        f = open(os.path.join(root_dir, 'user_data', 'data', 'online', 'article_w2v.pkl'), 'rb')
        article_vec_map = pickle.load(f)
        f.close()

    log.info('计算 W2V 相似度特征（向量化优化）...')
    last_sim_w2v, sum_sim_w2v = compute_w2v_sim_features_vectorized(
        df_feature, user_item_dict, article_vec_map, num=2
    )
    df_feature['user_last_click_article_w2v_sim'] = last_sim_w2v
    df_feature['user_click_article_w2w_sim_sum_2'] = sum_sim_w2v

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 保存特征文件
    if mode == 'valid':
        output_dir = os.path.join(root_dir, 'user_data', 'data', 'offline')
        os.makedirs(output_dir, exist_ok=True)
        df_feature.to_pickle(os.path.join(output_dir, 'feature.pkl'))

    else:
        output_dir = os.path.join(root_dir, 'user_data', 'data', 'online')
        os.makedirs(output_dir, exist_ok=True)
        df_feature.to_pickle(os.path.join(output_dir, 'feature.pkl'))

