import argparse
import gc
import os
import random
import warnings

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from common.utils import Logger
from eval.evaluate import evaluate
from rank.utils import gen_sub

# 获取项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

seed = 2020
random.seed(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='lightgbm 排序')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
log_dir = os.path.join(root_dir, 'user_data', 'log')
os.makedirs(log_dir, exist_ok=True)
log = Logger(os.path.join(log_dir, logfile)).logger
log.info(f'lightgbm 排序，mode: {mode}')


def train_model(df_feature, df_query):
    df_train = df_feature[df_feature['label'].notnull()]
    df_test = df_feature[df_feature['label'].isnull()]

    del df_feature
    gc.collect()

    ycol = 'label'
    feature_names = list(
        filter(
            lambda x: x not in [ycol, 'created_at_datetime', 'click_datetime'],
            df_train.columns))
    feature_names.sort()

    model = lgb.LGBMClassifier(num_leaves=64,
                               max_depth=10,
                               learning_rate=0.05,
                               n_estimators=10000,
                               subsample=0.8,
                               feature_fraction=0.8,
                               reg_alpha=0.5,
                               reg_lambda=0.5,
                               random_state=seed,
                               importance_type='gain',
                               metric=None)

    oof = []
    prediction = df_test[['user_id', 'article_id']]
    prediction['pred'] = 0
    df_importance_list = []

    # 创建模型保存目录
    model_dir = os.path.join(root_dir, 'user_data', 'model')
    os.makedirs(model_dir, exist_ok=True)

    # 训练模型
    kfold = GroupKFold(n_splits=5)
    for fold_id, (trn_idx, val_idx) in enumerate(
            kfold.split(df_train[feature_names], df_train[ycol],
                        df_train['user_id'])):
        X_train = df_train.iloc[trn_idx][feature_names]
        Y_train = df_train.iloc[trn_idx][ycol]

        X_val = df_train.iloc[val_idx][feature_names]
        Y_val = df_train.iloc[val_idx][ycol]

        # 检查模型是否已存在（断点续训）
        model_path = os.path.join(model_dir, f'lgb{fold_id}.pkl')
        if os.path.exists(model_path):
            log.info(f'跳过 Fold_{fold_id + 1}，模型已存在: {model_path}')
            # 加载已存在的模型用于预测
            lgb_model = joblib.load(model_path)
            
            # 生成验证集预测
            pred_val = lgb_model.predict_proba(
                X_val, num_iteration=lgb_model.best_iteration_)[:, 1]
            df_oof = df_train.iloc[val_idx][['user_id', 'article_id', ycol]].copy()
            df_oof['pred'] = pred_val
            oof.append(df_oof)
            
            # 生成测试集预测
            pred_test = lgb_model.predict_proba(
                df_test[feature_names], num_iteration=lgb_model.best_iteration_)[:, 1]
            prediction['pred'] += pred_test / 5
            
            # 特征重要性
            df_importance = pd.DataFrame({
                'feature_name': feature_names,
                'importance': lgb_model.feature_importances_,
            })
            df_importance_list.append(df_importance)
            continue  # 跳过训练，继续下一折

        log.debug(
            f'\nFold_{fold_id + 1} Training ================================\n'
        )

        # 新版本LightGBM使用callbacks替代verbose
        from lightgbm import early_stopping, log_evaluation
        callbacks = [
            early_stopping(stopping_rounds=100),
            log_evaluation(period=100)
        ]
        lgb_model = model.fit(X_train,
                              Y_train,
                              eval_set=[(X_train, Y_train), (X_val, Y_val)],
                              eval_names=['train', 'valid'],
                              callbacks=callbacks)

        pred_val = lgb_model.predict_proba(
            X_val, num_iteration=lgb_model.best_iteration_)[:, 1]
        df_oof = df_train.iloc[val_idx][['user_id', 'article_id', ycol]].copy()
        df_oof['pred'] = pred_val
        oof.append(df_oof)

        pred_test = lgb_model.predict_proba(
            df_test[feature_names], num_iteration=lgb_model.best_iteration_)[:,
                                                                             1]
        prediction['pred'] += pred_test / 5

        df_importance = pd.DataFrame({
            'feature_name':
            feature_names,
            'importance':
            lgb_model.feature_importances_,
        })
        df_importance_list.append(df_importance)

        # 保存模型
        joblib.dump(lgb_model, model_path)
        log.info(f'Fold_{fold_id + 1} 模型已保存: {model_path}')

    # 特征重要性
    df_importance = pd.concat(df_importance_list)
    df_importance = df_importance.groupby([
        'feature_name'
    ])['importance'].agg('mean').sort_values(ascending=False).reset_index()
    log.debug(f'importance: {df_importance}')

    # 生成线下
    df_oof = pd.concat(oof)
    df_oof.sort_values(['user_id', 'pred'],
                       inplace=True,
                       ascending=[True, False])
    log.debug(f'df_oof.head: {df_oof.head()}')

    # 计算相关指标
    total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
        df_oof, total)
    log.debug(
        f'{hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
    )

    # 生成提交文件
    df_sub = gen_sub(prediction)
    df_sub.sort_values(['user_id'], inplace=True)
    pred_dir = os.path.join(root_dir, 'prediction_result')
    os.makedirs(pred_dir, exist_ok=True)
    df_sub.to_csv(os.path.join(pred_dir, 'result.csv'), index=False)


def online_predict(df_test):
    ycol = 'label'
    feature_names = list(
        filter(
            lambda x: x not in [ycol, 'created_at_datetime', 'click_datetime'],
            df_test.columns))
    feature_names.sort()

    prediction = df_test[['user_id', 'article_id']]
    prediction['pred'] = 0

    model_dir = os.path.join(root_dir, 'user_data', 'model')
    for fold_id in tqdm(range(5)):
        model = joblib.load(os.path.join(model_dir, f'lgb{fold_id}.pkl'))
        pred_test = model.predict_proba(df_test[feature_names])[:, 1]
        prediction['pred'] += pred_test / 5

    # 生成提交文件
    df_sub = gen_sub(prediction)
    df_sub.sort_values(['user_id'], inplace=True)
    pred_dir = os.path.join(root_dir, 'prediction_result')
    os.makedirs(pred_dir, exist_ok=True)
    df_sub.to_csv(os.path.join(pred_dir, 'result.csv'), index=False)


if __name__ == '__main__':
    if mode == 'valid':
        df_feature = pd.read_pickle(os.path.join(root_dir, 'user_data', 'data', 'offline', 'feature.pkl'))
        df_query = pd.read_pickle(os.path.join(root_dir, 'user_data', 'data', 'offline', 'query.pkl'))

        for f in df_feature.select_dtypes('object').columns:
            lbl = LabelEncoder()
            df_feature[f] = lbl.fit_transform(df_feature[f].astype(str))

        train_model(df_feature, df_query)
    else:
        df_feature = pd.read_pickle(os.path.join(root_dir, 'user_data', 'data', 'online', 'feature.pkl'))
        online_predict(df_feature)

