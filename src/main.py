"""
基线推荐系统主入口
"""

import time
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.data_loader import DataLoader
from src.services.recommender import BaselineRecommender
from src.services.evaluator import Evaluator
from src.services.output_writer import OutputWriter
from src.services.model_manager import ModelManager
from src.algorithms.itemcf import ItemCF
from src.algorithms.usercf import UserCF
from src.algorithms.embedding_based import EmbeddingRecommender
from src.utils.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, ARTICLES_PATH, 
    ARTICLES_EMB_PATH, setup_random_seed, TOP_K, TEST_USER_LIMIT
)
from src.utils.logger import logger
from src.utils.exceptions import BaselineRecommenderError
from src.utils.progress import show_progress
from src.utils.matrix_builder import build_user_item_matrix


def run_baseline_recommender(
    train_data_path: str = TRAIN_DATA_PATH,
    test_data_path: str = TEST_DATA_PATH,
    articles_path: str = ARTICLES_PATH,
    embeddings_path: str = ARTICLES_EMB_PATH,
    output_path: str = 'data/submit.csv',
    use_cache: bool = True,
    save_models: bool = True,
    test_user_limit: int = None,
    use_local_validation: bool = True
) -> float:
    """
    运行基线推荐系统主流程
    
    Args:
        train_data_path: 训练集文件路径
        test_data_path: 测试集文件路径
        articles_path: 文章信息文件路径
        embeddings_path: 文章embedding文件路径
        output_path: 输出文件路径
        use_cache: 是否使用缓存的模型
        save_models: 是否保存训练的模型
        test_user_limit: 测试用户数量限制（None表示使用所有用户，用于快速测试）
        use_local_validation: 是否使用本地验证（从训练数据中分割验证集，默认True）
                              True: 从训练数据中分割，每个用户最后一条点击作为验证
                              False: 使用外部测试集（用于最终提交）
        
    Returns:
        MRR评估指标值
    """
    start_time = time.time()
    
    try:
        logger.info("=" * 60)
        logger.info("Baseline Recommender System - Starting")
        logger.info("=" * 60)
        
        # 如果没有指定test_user_limit，使用配置文件中的默认值
        if test_user_limit is None:
            test_user_limit = TEST_USER_LIMIT
        
        # 如果设置了测试用户限制，显示提示信息
        if test_user_limit is not None and test_user_limit > 0:
            logger.info(f"⚠️  TEST MODE ENABLED: Will process only first {test_user_limit} users")
            logger.info(f"   Set test_user_limit=None to process all users")
        
        # 设置随机种子
        setup_random_seed()
        logger.info("Random seed configured for reproducibility")
        
        # 1. 数据加载
        logger.info("\n[Step 1] Loading data...")
        loader = DataLoader()
        
        if use_local_validation:
            # 本地验证模式：从训练数据中分割出验证集
            logger.info("Using local validation: splitting train data into train/validation sets")
            
            # 先尝试加载已保存的分割数据
            train_data, validation_data = loader.load_split_data()
            
            if train_data is None or validation_data is None:
                # 如果不存在，则进行分割
                logger.info("Split data not found, splitting from full training data...")
                full_train_data = loader.load_train_data(train_data_path)
                train_data, validation_data = loader.split_train_validation(full_train_data)
                # 保存分割后的数据到文件
                try:
                    loader.save_split_data(train_data, validation_data)
                except Exception as e:
                    logger.warning(f"Failed to save split data: {str(e)}")
            else:
                logger.info("✓ Using cached split data (loaded from files)")
            
            # 不使用外部测试集，使用分割出的验证集
            test_data = validation_data
            logger.info("⚠️  LOCAL VALIDATION MODE: Using split validation set instead of external test set")
        else:
            # 使用外部测试集模式（用于最终提交）
            train_data = loader.load_train_data(train_data_path)
            test_data = loader.load_test_data(test_data_path)
            logger.info("Using external test set for final submission")
        
        articles = loader.load_articles(articles_path)
        embeddings = loader.load_article_embeddings(embeddings_path)
        logger.info("Data loading completed")
        
        # 2. 模型训练或加载
        logger.info("\n[Step 2] Loading or training recommendation models...")
        model_manager = ModelManager()
        
        # 尝试加载已保存的模型
        if use_cache:
            logger.info("Checking for saved models...")
            loaded_models = model_manager.load_all_models()
            if loaded_models:
                logger.info("Found saved models, loading...")
                itemcf = loaded_models['itemcf']
                usercf = loaded_models['usercf']
                embedding_rec = loaded_models['embedding']
                popular_articles = loaded_models.get('popular_articles')
                logger.info("Models loaded successfully from cache")
            else:
                logger.info("No saved models found, training new models...")
                loaded_models = None
        else:
            loaded_models = None
        
        # 如果模型未加载，则训练新模型
        if loaded_models is None:
            # 优化：先构建共享的user-item矩阵，避免重复计算
            # 使用articles中的所有文章ID（与ItemCF一致），确保索引匹配
            logger.info("Building shared user-item interaction matrix (for ItemCF and UserCF)...")
            article_ids_for_matrix = articles['article_id'].unique().tolist()
            shared_user_item_matrix, shared_user_id_to_index, shared_article_id_to_index = \
                build_user_item_matrix(train_data, article_ids=article_ids_for_matrix)
            
            logger.info("Training ItemCF model...")
            itemcf = ItemCF()
            item_cooccurrence = itemcf.train(
                train_data, articles,
                user_item_matrix=shared_user_item_matrix,
                user_id_to_index=shared_user_id_to_index,
                article_id_to_index=shared_article_id_to_index
            )  # 返回共现矩阵，相似度按需计算
            
            logger.info("Training UserCF model...")
            usercf = UserCF()
            user_item_matrix = usercf.train(
                train_data,
                user_item_matrix=shared_user_item_matrix,
                user_id_to_index=shared_user_id_to_index,
                article_id_to_index=shared_article_id_to_index
            )  # 返回user_item_matrix，相似度按需计算
            
            logger.info("Training Embedding model...")
            embedding_rec = EmbeddingRecommender()
            article_ids = articles['article_id'].tolist()
            embedding_rec.compute_similarity(embeddings, article_ids)  # 不再返回相似度矩阵
            
            # 获取热门文章
            logger.info("Computing popular articles...")
            article_counts = train_data['click_article_id'].value_counts()
            popular_articles = article_counts.head(100).index.tolist()
            
            # 保存模型
            if save_models:
                logger.info("Saving trained models...")
                model_manager.save_all_models(itemcf, usercf, embedding_rec, popular_articles)
                logger.info("Models saved successfully")
        
        # 设置推荐器
        recommender = BaselineRecommender()
        recommender.itemcf = itemcf
        recommender.usercf = usercf
        recommender.embedding_rec = embedding_rec
        if loaded_models and popular_articles:
            recommender.popular_articles = popular_articles
        else:
            recommender.get_popular_articles(train_data)
        
        logger.info("Model preparation completed")
        
        # 3. 生成推荐
        logger.info("\n[Step 3] Generating recommendations...")
        recommendations = {}
        test_users_all = test_data['user_id'].unique()
        total_users_all = len(test_users_all)
        
        # 如果设置了测试用户限制，只使用前N个用户
        if test_user_limit is not None and test_user_limit > 0:
            test_users = test_users_all[:test_user_limit]
            total_users = len(test_users)
            logger.info(f"⚠️  TEST MODE: Limited to {total_users} users for quick testing")
            logger.info(f"   (Original test set has {total_users_all} users)")
        else:
            test_users = test_users_all
            total_users = total_users_all
            logger.info(f"Generating recommendations for {total_users} users (full test set)...")
        
        # 优化：预先计算用户历史点击（避免重复查询）
        logger.info("Pre-computing user histories...")
        user_histories = {}
        # 确保用户ID类型一致（转换为字符串以便匹配）
        train_data_user_id_str = train_data['user_id'].astype(str)
        for user_id in test_users:
            user_id_str = str(user_id)
            user_history = train_data[train_data_user_id_str == user_id_str]['click_article_id'].tolist()
            # 确保文章ID都是字符串类型
            user_history = [str(aid) for aid in user_history]
            user_histories[user_id] = user_history
        logger.info(f"Pre-computed histories for {len(user_histories)} users")
        # 统计有历史记录的用户数量
        users_with_history = sum(1 for h in user_histories.values() if len(h) > 0)
        logger.info(f"Users with history: {users_with_history}/{len(user_histories)} ({100*users_with_history/len(user_histories):.1f}%)")
        
        logger.info("Generating recommendations...")
        # 生成推荐（带进度显示）
        for i, user_id in enumerate(show_progress(test_users, desc="Generating recommendations", total=total_users)):
            user_history = user_histories.get(user_id, [])
            
            # 生成推荐
            try:
                recs = recommender.recommend_for_user(user_id, user_history, train_data, TOP_K)
                recommendations[user_id] = recs
            except Exception as e:
                logger.warning(f"Failed to generate recommendations for user {user_id}: {str(e)}")
                # 使用热门文章作为后备
                if recommender.popular_articles:
                    recommendations[user_id] = recommender.popular_articles[:TOP_K]
                else:
                    recommendations[user_id] = []
        
        logger.info(f"Recommendations generated for {len(recommendations)} users")
        
        # 4. 评估
        logger.info("\n[Step 4] Evaluating recommendations...")
        evaluator = Evaluator()
        # 如果限制了测试用户，只评估这些用户的测试数据
        if test_user_limit is not None and test_user_limit > 0:
            test_data_filtered = test_data[test_data['user_id'].isin(test_users)]
            evaluation_result = evaluator.evaluate(recommendations, test_data_filtered)
        else:
            evaluation_result = evaluator.evaluate(recommendations, test_data)
        mrr = evaluation_result['mrr']
        logger.info(f"Evaluation completed: MRR = {mrr:.4f}")
        if test_user_limit is not None and test_user_limit > 0:
            logger.info(f"⚠️  This is a TEST result based on {total_users} users only")
        
        # 5. 输出结果
        logger.info("\n[Step 5] Writing submission file...")
        writer = OutputWriter()
        writer.write_submission(recommendations, output_path)
        logger.info("Submission file written")
        
        # 计算总时间
        elapsed_time = time.time() - start_time
        logger.info(f"\nTotal processing time: {elapsed_time:.2f} seconds ({elapsed_time/3600:.2f} hours)")
        
        logger.info("=" * 60)
        logger.info("Baseline Recommender System - Completed Successfully")
        logger.info("=" * 60)
        
        return mrr
        
    except BaselineRecommenderError as e:
        logger.error(f"Baseline recommender error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Baseline Recommender System')
    parser.add_argument('--test-users', type=int, default=None,
                        help='Limit number of test users for quick testing (e.g., --test-users 100)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable model caching (force retraining)')
    parser.add_argument('--no-save', action='store_true',
                        help='Disable model saving')
    parser.add_argument('--output', type=str, default='data/submit.csv',
                        help='Output file path (default: data/submit.csv)')
    parser.add_argument('--external-test', action='store_true',
                        help='Use external test set instead of local validation (for final submission)')
    
    args = parser.parse_args()
    
    try:
        mrr = run_baseline_recommender(
            test_user_limit=args.test_users,
            use_cache=not args.no_cache,
            save_models=not args.no_save,
            output_path=args.output,
            use_local_validation=not args.external_test
        )
        print(f"\nFinal MRR Score: {mrr:.4f}")
        if args.test_users:
            print(f"⚠️  Note: This result is based on {args.test_users} users only (test mode)")
        if not args.external_test:
            print(f"ℹ️  Note: Using local validation (split from training data)")
        else:
            print(f"ℹ️  Note: Using external test set (for final submission)")
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

