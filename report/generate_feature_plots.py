"""
Generate visualization charts for dataset features
"""
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import platform

from matplotlib import font_manager

def set_ch_font():
    # In English output mode, skip Chinese font settings, fallback to usual sans-serif.
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    return font_manager.FontProperties()

# Use default font properties for English
FONT_PROP = set_ch_font()

def main():
    # Get project root
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root_dir, 'report', 'assets')
    os.makedirs(output_dir, exist_ok=True)

    # Load feature data
    feature_file = os.path.join(root_dir, 'user_data', 'data', 'offline', 'feature.pkl')
    if os.path.exists(feature_file):
        df_feature = pd.read_pickle(feature_file)
        print(f"Feature data shape: {df_feature.shape}")
    else:
        print(f"Feature file not found: {feature_file}")
        exit(1)

    # Load article data
    article_file = os.path.join(root_dir, 'data', 'articles.csv')
    if os.path.exists(article_file):
        df_article = pd.read_csv(article_file)
        if 'created_at_ts' in df_article.columns:
            df_article['created_at_ts'] = df_article['created_at_ts'] / 1000
            df_article['created_at_ts'] = df_article['created_at_ts'].astype('int')
    else:
        print(f"Article file not found: {article_file}")
        df_article = None

    # 1. Article word count distribution
    if 'words_count' in df_feature.columns:
        plt.figure(figsize=(10, 6))
        words_data = df_feature['words_count'].dropna()
        plt.hist(words_data, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Article Word Count', fontsize=12, fontproperties=FONT_PROP)
        plt.ylabel('Frequency', fontsize=12, fontproperties=FONT_PROP)
        plt.title('Distribution of Article Word Count', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'words_count_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Article word count distribution: mean={words_data.mean():.2f}, median={words_data.median():.2f}")

    # 2. Article category distribution
    if 'category_id' in df_feature.columns:
        plt.figure(figsize=(12, 6))
        category_counts = df_feature['category_id'].value_counts().head(20)
        plt.bar(range(len(category_counts)), category_counts.values, edgecolor='black', alpha=0.7)
        plt.xlabel('Article Category ID', fontsize=12, fontproperties=FONT_PROP)
        plt.ylabel('Article Count', fontsize=12, fontproperties=FONT_PROP)
        plt.title('Top 20 Article Category Distribution', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        plt.xticks(range(len(category_counts)), category_counts.index, rotation=45, fontproperties=FONT_PROP)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'category_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Similarity score distribution
    if 'sim_score' in df_feature.columns:
        plt.figure(figsize=(10, 6))
        sim_data = df_feature['sim_score'].dropna()
        plt.hist(sim_data, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Similarity Score', fontsize=12, fontproperties=FONT_PROP)
        plt.ylabel('Frequency', fontsize=12, fontproperties=FONT_PROP)
        plt.title('Distribution of Recall Similarity Scores', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sim_score_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Similarity score: mean={sim_data.mean():.4f}, median={sim_data.median():.4f}")

    # 4. User last click timestamp difference distribution
    if 'user_last_click_timestamp_diff' in df_feature.columns:
        plt.figure(figsize=(10, 6))
        time_diff = df_feature['user_last_click_timestamp_diff'].dropna()
        # Filter outliers
        time_diff = time_diff[(time_diff >= -1e10) & (time_diff <= 1e10)]
        plt.hist(time_diff, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('User Last Click Time Difference (s)', fontsize=12, fontproperties=FONT_PROP)
        plt.ylabel('Frequency', fontsize=12, fontproperties=FONT_PROP)
        plt.title('Distribution of User Last Click Time Difference', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'user_last_click_timestamp_diff.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 5. ItemCF similarity feature distribution
    if 'user_last_click_article_itemcf_sim' in df_feature.columns:
        plt.figure(figsize=(10, 6))
        itemcf_sim = df_feature['user_last_click_article_itemcf_sim'].dropna()
        itemcf_sim = itemcf_sim[itemcf_sim > 0]  # Only show positive similarities
        plt.hist(itemcf_sim, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('ItemCF Similarity', fontsize=12, fontproperties=FONT_PROP)
        plt.ylabel('Frequency', fontsize=12, fontproperties=FONT_PROP)
        plt.title('Distribution of ItemCF Similarity (User Last Click vs Candidate)', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'itemcf_sim_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 6. User click count distribution
    if 'user_id_cnt' in df_feature.columns:
        plt.figure(figsize=(10, 6))
        user_cnt = df_feature['user_id_cnt'].dropna()
        # Show up to 95th percentile to avoid long tail
        user_cnt_filtered = user_cnt[user_cnt <= user_cnt.quantile(0.95)]
        plt.hist(user_cnt_filtered, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('User Click Count', fontsize=12, fontproperties=FONT_PROP)
        plt.ylabel('Frequency', fontsize=12, fontproperties=FONT_PROP)
        plt.title('Distribution of User Click Count (within 95th percentile)', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'user_click_count_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 7. Article click count distribution
    if 'article_id_cnt' in df_feature.columns:
        plt.figure(figsize=(10, 6))
        article_cnt = df_feature['article_id_cnt'].dropna()
        # Show up to 95th percentile
        article_cnt_filtered = article_cnt[article_cnt <= article_cnt.quantile(0.95)]
        plt.hist(article_cnt_filtered, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Article Click Count', fontsize=12, fontproperties=FONT_PROP)
        plt.ylabel('Frequency', fontsize=12, fontproperties=FONT_PROP)
        plt.title('Distribution of Article Click Count (within 95th percentile)', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'article_click_count_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 8. Label distribution (positive/negative sample ratio)
    if 'label' in df_feature.columns:
        plt.figure(figsize=(8, 6))
        label_counts = df_feature['label'].value_counts()
        labels = ['Negative (0)', 'Positive (1)']
        values = [label_counts.get(0, 0), label_counts.get(1, 0)]
        plt.bar(labels, values,
                color=['#ff7f7f', '#7fbf7f'], edgecolor='black', alpha=0.7)
        plt.ylabel('Sample Count', fontsize=12, fontproperties=FONT_PROP)
        plt.title('Distribution of Positive/Negative Samples', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        # Add value labels
        for i, v in enumerate(values):
            plt.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=11, fontproperties=FONT_PROP)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'label_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        if values[1] != 0:
            ratio = values[0] / values[1]
            print(f"Positive/Negative Sample Ratio: {values[0]:,} : {values[1]:,} = {ratio:.2f}:1")
        else:
            print(f"Positive/Negative Sample Ratio: {values[0]:,} : {values[1]:,} (denominator is 0)")

    # 9. Article creation time distribution
    if 'created_at_ts' in df_feature.columns:
        plt.figure(figsize=(12, 6))
        created_time = df_feature['created_at_ts'].dropna()
        # Convert to days since min time for better visualization
        min_time = created_time.min()
        days_since_min = (created_time - min_time) / (24 * 3600)
        plt.hist(days_since_min, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Days Since Minimum Creation Time', fontsize=12, fontproperties=FONT_PROP)
        plt.ylabel('Frequency', fontsize=12, fontproperties=FONT_PROP)
        plt.title('Distribution of Article Creation Time', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'article_creation_time_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 10. Binetwork similarity distribution
    if 'user_last_click_article_binetwork_sim' in df_feature.columns:
        plt.figure(figsize=(10, 6))
        binetwork_sim = df_feature['user_last_click_article_binetwork_sim'].dropna()
        binetwork_sim = binetwork_sim[binetwork_sim > 0]
        plt.hist(binetwork_sim, bins=50, edgecolor='black', alpha=0.7, color='#ff9999')
        plt.xlabel('Binetwork Similarity', fontsize=12, fontproperties=FONT_PROP)
        plt.ylabel('Frequency', fontsize=12, fontproperties=FONT_PROP)
        plt.title('Distribution of Binetwork Similarity', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'binetwork_sim_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 11. W2V similarity distribution
    if 'user_last_click_article_w2v_sim' in df_feature.columns:
        plt.figure(figsize=(10, 6))
        w2v_sim = df_feature['user_last_click_article_w2v_sim'].dropna()
        w2v_sim = w2v_sim[w2v_sim > -1]  # Filter invalid values
        plt.hist(w2v_sim, bins=50, edgecolor='black', alpha=0.7, color='#99ccff')
        plt.xlabel('W2V Similarity (Cosine)', fontsize=12, fontproperties=FONT_PROP)
        plt.ylabel('Frequency', fontsize=12, fontproperties=FONT_PROP)
        plt.title('Distribution of W2V Similarity', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'w2v_sim_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 12. User-article words count comparison (positive vs negative)
    if 'words_count' in df_feature.columns and 'label' in df_feature.columns:
        plt.figure(figsize=(10, 6))
        pos_words = df_feature[df_feature['label'] == 1]['words_count'].dropna()
        neg_words = df_feature[df_feature['label'] == 0]['words_count'].dropna()
        # Sample for visualization if too large
        if len(neg_words) > 100000:
            neg_words = neg_words.sample(n=100000, random_state=42)
        plt.hist(neg_words, bins=50, alpha=0.5, label='Negative (0)', color='#ff7f7f', edgecolor='black')
        plt.hist(pos_words, bins=50, alpha=0.5, label='Positive (1)', color='#7fbf7f', edgecolor='black')
        plt.xlabel('Article Word Count', fontsize=12, fontproperties=FONT_PROP)
        plt.ylabel('Frequency', fontsize=12, fontproperties=FONT_PROP)
        plt.title('Word Count Distribution: Positive vs Negative Samples', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'words_count_pos_neg_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 13. Similarity score comparison (positive vs negative)
    if 'sim_score' in df_feature.columns and 'label' in df_feature.columns:
        plt.figure(figsize=(10, 6))
        pos_sim = df_feature[df_feature['label'] == 1]['sim_score'].dropna()
        neg_sim = df_feature[df_feature['label'] == 0]['sim_score'].dropna()
        # Sample for visualization if too large
        if len(neg_sim) > 100000:
            neg_sim = neg_sim.sample(n=100000, random_state=42)
        plt.hist(neg_sim, bins=50, alpha=0.5, label='Negative (0)', color='#ff7f7f', edgecolor='black')
        plt.hist(pos_sim, bins=50, alpha=0.5, label='Positive (1)', color='#7fbf7f', edgecolor='black')
        plt.xlabel('Similarity Score', fontsize=12, fontproperties=FONT_PROP)
        plt.ylabel('Frequency', fontsize=12, fontproperties=FONT_PROP)
        plt.title('Similarity Score Distribution: Positive vs Negative Samples', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sim_score_pos_neg_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Positive samples sim_score mean: {pos_sim.mean():.4f}, Negative samples sim_score mean: {neg_sim.mean():.4f}")

    # 14. User click time interval distribution
    if 'user_id_click_diff_mean' in df_feature.columns:
        plt.figure(figsize=(10, 6))
        click_diff = df_feature['user_id_click_diff_mean'].dropna()
        # Convert to hours and filter outliers
        click_diff_hours = click_diff / 3600
        click_diff_hours = click_diff_hours[(click_diff_hours >= 0) & (click_diff_hours <= 168)]  # 0-7 days
        plt.hist(click_diff_hours, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Average Click Interval (hours)', fontsize=12, fontproperties=FONT_PROP)
        plt.ylabel('Frequency', fontsize=12, fontproperties=FONT_PROP)
        plt.title('Distribution of User Average Click Interval', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'user_click_interval_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 15. User clicked article words count mean distribution
    if 'user_clicked_article_words_count_mean' in df_feature.columns:
        plt.figure(figsize=(10, 6))
        words_mean = df_feature['user_clicked_article_words_count_mean'].dropna()
        plt.hist(words_mean, bins=50, edgecolor='black', alpha=0.7, color='#ffcc99')
        plt.xlabel('Average Words Count of User Clicked Articles', fontsize=12, fontproperties=FONT_PROP)
        plt.ylabel('Frequency', fontsize=12, fontproperties=FONT_PROP)
        plt.title('Distribution of User Average Clicked Article Word Count', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'user_avg_words_count_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 16. Words count difference distribution
    if 'user_last_click_words_count_diff' in df_feature.columns:
        plt.figure(figsize=(10, 6))
        words_diff = df_feature['user_last_click_words_count_diff'].dropna()
        # Filter extreme outliers
        q1, q99 = words_diff.quantile([0.01, 0.99])
        words_diff_filtered = words_diff[(words_diff >= q1) & (words_diff <= q99)]
        plt.hist(words_diff_filtered, bins=50, edgecolor='black', alpha=0.7, color='#cc99ff')
        plt.xlabel('Words Count Difference', fontsize=12, fontproperties=FONT_PROP)
        plt.ylabel('Frequency', fontsize=12, fontproperties=FONT_PROP)
        plt.title('Distribution of Words Count Difference (Candidate - Last Clicked)', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No difference')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'words_count_diff_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 17. Category distribution by positive/negative samples
    if 'category_id' in df_feature.columns and 'label' in df_feature.columns:
        plt.figure(figsize=(14, 6))
        top_categories = df_feature['category_id'].value_counts().head(15).index
        df_top = df_feature[df_feature['category_id'].isin(top_categories)]
        category_label = df_top.groupby(['category_id', 'label']).size().unstack(fill_value=0)
        category_label = category_label.reindex(top_categories)
        x = np.arange(len(top_categories))
        width = 0.35
        pos_counts = category_label.get(1, pd.Series(0, index=top_categories))
        neg_counts = category_label.get(0, pd.Series(0, index=top_categories))
        plt.bar(x - width/2, neg_counts.values, width, label='Negative (0)', color='#ff7f7f', alpha=0.7, edgecolor='black')
        plt.bar(x + width/2, pos_counts.values, width, label='Positive (1)', color='#7fbf7f', alpha=0.7, edgecolor='black')
        plt.xlabel('Category ID', fontsize=12, fontproperties=FONT_PROP)
        plt.ylabel('Count', fontsize=12, fontproperties=FONT_PROP)
        plt.title('Category Distribution: Positive vs Negative Samples (Top 15)', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        plt.xticks(x, top_categories, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'category_pos_neg_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 18. Feature correlation heatmap (top features)
    if 'label' in df_feature.columns:
        # Select numeric features for correlation
        numeric_cols = df_feature.select_dtypes(include=[np.number]).columns.tolist()
        # Remove ID columns and label
        exclude_cols = ['user_id', 'article_id', 'label', 'created_at_datetime']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        # Select top 15 features by variance
        if len(numeric_cols) > 15:
            variances = df_feature[numeric_cols].var().sort_values(ascending=False)
            top_features = variances.head(15).index.tolist()
        else:
            top_features = numeric_cols[:15]
        
        if len(top_features) > 0:
            plt.figure(figsize=(12, 10))
            corr_matrix = df_feature[top_features].corr()
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
            plt.title('Feature Correlation Heatmap (Top 15 Features)', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()

    print(f"\nAll charts have been saved to: {output_dir}")

if __name__ == '__main__':
    main()
