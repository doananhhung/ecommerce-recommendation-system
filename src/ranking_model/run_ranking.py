import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import config
from src.ranking_model.lgbm_train import RankingTrainer

def run_ranking_training():
    print("1. Loading Data...")
    df_interactions = pd.read_parquet(config.LABELED_SESSIONS_PATH)
    user_features = pd.read_parquet(config.USER_FEATURES_PATH)
    item_features = pd.read_parquet(config.ITEM_FEATURES_PATH)
    
    print("2. Joining Features...")
    df = df_interactions.merge(user_features, on='user_id', how='left')
    df = df.merge(item_features, on='product_id', how='left')
    
    # Fill NaN cho các đặc trưng (nếu có)
    df.fillna(0, inplace=True)
    
    print("3. Preparing Train/Test Split...")
    features = [
        'user_total_interactions', 'user_total_sessions', 
        'item_total_interactions', 'item_unique_users', 'item_avg_price'
    ]
    target = 'label'
    
    df_train, df_test = train_test_split(df, test_size=config.RANKING_TEST_SIZE, random_state=42)
    X_train = df_train[features]
    y_train = df_train[target]
    X_test = df_test[features]
    y_test = df_test[target]
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    print("4. Training LightGBM Model...")
    trainer = RankingTrainer(learning_rate=config.RANKING_LEARNING_RATE, num_leaves=config.RANKING_NUM_LEAVES)
    preds = trainer.train(X_train, y_train, X_test, y_test, num_boost_round=config.RANKING_NUM_BOOST_ROUND)
    
    print("5. Evaluating Model...")
    query_groups = df_test['user_id'].values
    trainer.evaluate(y_test.values, preds, query_groups=query_groups, k=10)
    
    print("6. Saving Model...")
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    trainer.save_model(config.RANKER_MODEL_PATH)
    print("Ranking Stage Finished Successfully!")

if __name__ == "__main__":
    run_ranking_training()