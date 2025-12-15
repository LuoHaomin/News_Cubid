import os
import pickle
from random import sample

import multitasking
import pandas as pd
from tqdm import tqdm

from common.utils import max_threads

# 获取项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@multitasking.task
def gen_sub_multitasking(test_users, prediction, all_articles, worker_id):
    lines = []

    for test_user in tqdm(test_users):
        g = prediction[prediction['user_id'] == test_user]
        g = g.head(5)
        items = g['article_id'].values.tolist()

        if len(set(items)) < 5:
            buchong = all_articles - set(items)
            buchong = sample(buchong, 5 - len(set(items)))
            items += buchong

        assert len(set(items)) == 5

        lines.append([test_user] + items)

    tmp_dir = os.path.join(root_dir, 'user_data', 'tmp', 'sub')
    os.makedirs(tmp_dir, exist_ok=True)

    with open(os.path.join(tmp_dir, f'{worker_id}.pkl'), 'wb') as f:
        pickle.dump(lines, f)


def gen_sub(prediction):
    prediction.sort_values(['user_id', 'pred'],
                           inplace=True,
                           ascending=[True, False])

    all_articles = set(prediction['article_id'].values)

    sub_sample = pd.read_csv(os.path.join(root_dir, 'data', 'testA_click_log.csv'))
    test_users = sub_sample.user_id.unique()

    n_split = max_threads
    total = len(test_users)
    n_len = total // n_split

    # 清空临时文件夹
    tmp_dir = os.path.join(root_dir, 'user_data', 'tmp', 'sub')
    if os.path.exists(tmp_dir):
        for path, _, file_list in os.walk(tmp_dir):
            for file_name in file_list:
                os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = test_users[i:i + n_len]
        gen_sub_multitasking(part_users, prediction, all_articles, i)

    multitasking.wait_for_tasks()

    lines = []
    for path, _, file_list in os.walk(tmp_dir):
        for file_name in file_list:
            with open(os.path.join(path, file_name), 'rb') as f:
                line = pickle.load(f)
                lines += line

    df_sub = pd.DataFrame(lines)
    df_sub.columns = [
        'user_id', 'article_1', 'article_2', 'article_3', 'article_4',
        'article_5'
    ]
    return df_sub

