import argparse
import math
import os
import pickle
import random
import signal
import warnings
from collections import defaultdict
from random import shuffle

import multitasking
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from gensim.models import Word2Vec
from tqdm import tqdm

from common.utils import Logger, max_threads
from eval.evaluate import evaluate

# 获取项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

multitasking.set_max_threads(max_threads)
# Windows上使用thread引擎，避免pickle问题
import sys
if sys.platform == 'win32':
    multitasking.set_engine('thread')
else:
    multitasking.set_engine('process')
    signal.signal(signal.SIGINT, multitasking.killall)

seed = 2020
random.seed(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='w2v 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
log_dir = os.path.join(root_dir, 'user_data', 'log')
os.makedirs(log_dir, exist_ok=True)
log = Logger(os.path.join(log_dir, logfile)).logger
log.info(f'w2v 召回，mode: {mode}')


def word2vec(df_, f1, f2, model_path):
    df = df_.copy()
    tmp = df.groupby(f1, as_index=False)[f2].agg(
        {'{}_{}_list'.format(f1, f2): list})

    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]

    words = []
    for i in range(len(sentences)):
        x = [str(x) for x in sentences[i]]
        sentences[i] = x
        words += x

    if os.path.exists(f'{model_path}/w2v.m'):
        model = Word2Vec.load(f'{model_path}/w2v.m')
    else:
        # 新版本gensim使用vector_size替代size
        model = Word2Vec(sentences=sentences,
                         vector_size=256,
                         window=3,
                         min_count=1,
                         sg=1,
                         hs=0,
                         seed=seed,
                         negative=5,
                         workers=10,
                         epochs=1)
        model.save(f'{model_path}/w2v.m')

    article_vec_map = {}
    for word in set(words):
        # 新版本gensim使用model.wv访问词向量
        if word in model.wv:
            article_vec_map[int(word)] = model.wv[word]

    return article_vec_map


@multitasking.task
def recall(df_query, article_vec_map, article_index, user_item_dict,
           worker_id):
    data_list = []

    for user_id, item_id in tqdm(df_query.values):
        rank = defaultdict(int)

        interacted_items = user_item_dict[user_id]
        interacted_items = interacted_items[-1:]

        for item in interacted_items:
            article_vec = article_vec_map[item]

            item_ids, distances = article_index.get_nns_by_vector(
                article_vec, 100, include_distances=True)
            sim_scores = [2 - distance for distance in distances]

            for relate_item, wij in zip(item_ids, sim_scores):
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    rank[relate_item] += wij

        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:50]
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id

        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    df_data = pd.concat(data_list, sort=False)

    tmp_dir = os.path.join(root_dir, 'user_data', 'tmp', 'w2v')
    os.makedirs(tmp_dir, exist_ok=True)
    df_data.to_pickle(os.path.join(tmp_dir, '{}.pkl'.format(worker_id)))


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle(os.path.join(root_dir, 'user_data', 'data', 'offline', 'click.pkl'))
        df_query = pd.read_pickle(os.path.join(root_dir, 'user_data', 'data', 'offline', 'query.pkl'))

        data_dir = os.path.join(root_dir, 'user_data', 'data', 'offline')
        os.makedirs(data_dir, exist_ok=True)
        model_dir = os.path.join(root_dir, 'user_data', 'model', 'offline')
        os.makedirs(model_dir, exist_ok=True)

        w2v_file = os.path.join(data_dir, 'article_w2v.pkl')
        model_path = model_dir
    else:
        df_click = pd.read_pickle(os.path.join(root_dir, 'user_data', 'data', 'online', 'click.pkl'))
        df_query = pd.read_pickle(os.path.join(root_dir, 'user_data', 'data', 'online', 'query.pkl'))

        data_dir = os.path.join(root_dir, 'user_data', 'data', 'online')
        os.makedirs(data_dir, exist_ok=True)
        model_dir = os.path.join(root_dir, 'user_data', 'model', 'online')
        os.makedirs(model_dir, exist_ok=True)

        w2v_file = os.path.join(data_dir, 'article_w2v.pkl')
        model_path = model_dir

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    article_vec_map = word2vec(df_click, 'user_id', 'click_article_id',
                               model_path)
    f = open(w2v_file, 'wb')
    pickle.dump(article_vec_map, f)
    f.close()

    # 将 embedding 建立索引
    article_index = AnnoyIndex(256, 'angular')
    article_index.set_seed(2020)

    for article_id, emb in tqdm(article_vec_map.items()):
        article_index.add_item(article_id, emb)

    article_index.build(100)

    user_item_ = df_click.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).reset_index()
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))

    # 召回
    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_len = total // n_split

    # 清空临时文件夹
    tmp_dir = os.path.join(root_dir, 'user_data', 'tmp', 'w2v')
    if os.path.exists(tmp_dir):
        for path, _, file_list in os.walk(tmp_dir):
            for file_name in file_list:
                os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, article_vec_map, article_index, user_item_dict, i)

    multitasking.wait_for_tasks()
    log.info('合并任务')

    df_data = pd.DataFrame()
    tmp_dir = os.path.join(root_dir, 'user_data', 'tmp', 'w2v')
    for path, _, file_list in os.walk(tmp_dir):
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            df_data = pd.concat([df_data, df_temp], ignore_index=True)

    # 必须加，对其进行排序
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True,
                                             False]).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')

    # 计算召回指标
    if mode == 'valid':
        log.info(f'计算召回指标')

        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()

        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)

        log.debug(
            f'w2v: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )
    # 保存召回结果
    if mode == 'valid':
        output_dir = os.path.join(root_dir, 'user_data', 'data', 'offline')
        os.makedirs(output_dir, exist_ok=True)
        df_data.to_pickle(os.path.join(output_dir, 'recall_w2v.pkl'))
    else:
        output_dir = os.path.join(root_dir, 'user_data', 'data', 'online')
        os.makedirs(output_dir, exist_ok=True)
        df_data.to_pickle(os.path.join(output_dir, 'recall_w2v.pkl'))

