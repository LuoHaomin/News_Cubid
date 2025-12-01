import argparse
import os
import random
from random import sample

import pandas as pd
from tqdm import tqdm

from common.utils import Logger

random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='数据处理')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(root_dir, 'user_data', 'log')
os.makedirs(log_dir, exist_ok=True)
log = Logger(os.path.join(log_dir, logfile)).logger
log.info(f'数据处理，mode: {mode}')


def data_offline(df_train_click, df_test_click):
    # 获取所有唯一的用户ID
    train_users = df_train_click['user_id'].unique().tolist()
    # 随机采样出一部分样本
    n_val_users = min(50000, len(train_users))
    val_users = set(sample(train_users, n_val_users))
    log.debug(f'val_users num: {len(val_users)}')
    log.debug(f'total train users: {len(train_users)}')

    # 训练集用户 抽出行为数据最后一条作为线下验证集
    click_list = []
    valid_query_list = []

    groups = df_train_click.groupby(['user_id'])
    matched_count = 0
    for user_id, g in tqdm(groups):
        # 确保类型一致 - groupby 返回的可能是标量或元组
        if isinstance(user_id, tuple):
            user_id_val = user_id[0]
        else:
            user_id_val = user_id
        
        # 转换为与 val_users 中相同的类型
        try:
            user_id_val = type(list(val_users)[0])(user_id_val)
        except:
            pass
            
        if user_id_val in val_users:
            matched_count += 1
            valid_query = g.tail(1)
            valid_query_list.append(
                valid_query[['user_id', 'click_article_id']])

            train_click = g.head(g.shape[0] - 1)
            click_list.append(train_click)
        else:
            click_list.append(g)
    
    log.debug(f'Matched validation users: {matched_count}')

    df_train_click = pd.concat(click_list, sort=False)
    
    # 检查是否有验证查询
    if len(valid_query_list) > 0:
        df_valid_query = pd.concat(valid_query_list, sort=False)
    else:
        log.warning('No validation queries found, creating empty DataFrame')
        df_valid_query = pd.DataFrame(columns=['user_id', 'click_article_id'])

    test_users = df_test_click['user_id'].unique()
    test_query_list = []

    for user in tqdm(test_users):
        test_query_list.append([user, -1])

    df_test_query = pd.DataFrame(test_query_list,
                                 columns=['user_id', 'click_article_id'])

    df_query = pd.concat([df_valid_query, df_test_query],
                         sort=False).reset_index(drop=True)
    df_click = pd.concat([df_train_click, df_test_click],
                         sort=False).reset_index(drop=True)
    df_click = df_click.sort_values(['user_id',
                                     'click_timestamp']).reset_index(drop=True)

    log.debug(
        f'df_query shape: {df_query.shape}, df_click shape: {df_click.shape}')
    log.debug(f'{df_query.head()}')
    log.debug(f'{df_click.head()}')

    # 保存文件
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    user_data_dir = os.path.join(root_dir, 'user_data', 'data', 'offline')
    os.makedirs(user_data_dir, exist_ok=True)

    df_click.to_pickle(os.path.join(user_data_dir, 'click.pkl'))
    df_query.to_pickle(os.path.join(user_data_dir, 'query.pkl'))


def data_online(df_train_click, df_test_click):
    test_users = df_test_click['user_id'].unique()
    test_query_list = []

    for user in tqdm(test_users):
        test_query_list.append([user, -1])

    df_test_query = pd.DataFrame(test_query_list,
                                 columns=['user_id', 'click_article_id'])

    df_query = df_test_query
    df_click = pd.concat([df_train_click, df_test_click],
                         sort=False).reset_index(drop=True)
    df_click = df_click.sort_values(['user_id',
                                     'click_timestamp']).reset_index(drop=True)

    log.debug(
        f'df_query shape: {df_query.shape}, df_click shape: {df_click.shape}')
    log.debug(f'{df_query.head()}')
    log.debug(f'{df_click.head()}')

    # 保存文件
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    user_data_dir = os.path.join(root_dir, 'user_data', 'data', 'online')
    os.makedirs(user_data_dir, exist_ok=True)

    df_click.to_pickle(os.path.join(user_data_dir, 'click.pkl'))
    df_query.to_pickle(os.path.join(user_data_dir, 'query.pkl'))


if __name__ == '__main__':
    import sys
    import os
    # 获取项目根目录
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df_train_click = pd.read_csv(os.path.join(root_dir, 'data', 'train_click_log.csv'))
    df_test_click = pd.read_csv(os.path.join(root_dir, 'data', 'testA_click_log.csv'))

    log.debug(
        f'df_train_click shape: {df_train_click.shape}, df_test_click shape: {df_test_click.shape}'
    )

    if mode == 'valid':
        data_offline(df_train_click, df_test_click)
    else:
        data_online(df_train_click, df_test_click)

