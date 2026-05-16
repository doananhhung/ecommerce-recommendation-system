import pandas as pd
import os
from sklearn.model_selection import train_test_split
from lgbm_train import RankingTrainer

def run_ranking_training(data_dir: str, models_dir: str):
    print("1. Loading Data...")
    df_interactions = pd.read_parquet(os.path.join(data_dir, 'sessions/labeled_sessions.parquet'))
    user_features = pd.read_parquet(os.path.join(data_dir, 'feature_store/user_features.parquet'))
    item_features = pd.read_parquet(os.path.join(data_dir, 'feature_store/item_features.parquet'))
    
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
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    print("4. Training LightGBM Model...")
    trainer = RankingTrainer(learning_rate=0.05, num_leaves=31)
    preds = trainer.train(X_train, y_train, X_test, y_test, num_boost_round=100)
    
    print("5. Evaluating Model...")
    trainer.evaluate(y_test.values, preds, k=10)
    
    print("6. Saving Model...")
    trainer.save_model(os.path.join(models_dir, 'ranker_model.txt'))
    print("Ranking Stage Finished Successfully!")

if __name__ == "__main__":
    DATA_DIR = 'data/'
    MODELS_DIR = 'models_store/'
    run_ranking_training(DATA_DIR, MODELS_DIR)
