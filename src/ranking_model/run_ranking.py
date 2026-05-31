import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import config
from src.ranking_model.lgbm_train import RankingTrainer

def run_ranking_training():
    print("1. Loading Data...")
    df = pd.read_parquet(config.LABELED_SESSIONS_PATH)
    # user_features và item_features là snapshot phục vụ serving.
    # Trong lúc training, df_interactions đã chứa point-in-time features.
    # Việc merge ở đây không những làm mất feature (sinh ra _x, _y) mà còn gây rò rỉ dữ liệu (data leakage).
    
    # Xử lý missing values đúng chuẩn thay vì fillna(0) toàn bộ
    if 'category_code' in df.columns:
        df['category_code'] = df['category_code'].fillna('<UNKNOWN>').astype('category')
    if 'brand' in df.columns:
        df['brand'] = df['brand'].fillna('<UNKNOWN>').astype('category')
    
    # Fill số 0 cho các cột số bị thiếu
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    print("3. Preparing Train/Test Split (Time-based)...")
    # Dynamically load features from ProjectConfig
    features = []
    for col in config.NUMERICAL_FEATURES:
        if col in df.columns:
            features.append(col)
    for col in config.CATEGORICAL_FEATURES:
        if col in df.columns:
            features.append(col)
            
    target = 'label'
    
    # TIME-BASED SPLITTING (Chống rò rỉ dữ liệu / Time Travel)
    # Sắp xếp theo thời gian và cắt % thay vì lấy ngẫu nhiên
    if 'event_time' in df.columns:
        df = df.sort_values('event_time').reset_index(drop=True)
    
    split_idx = int(len(df) * (1 - config.RANKING_TEST_SIZE))
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    
    X_train = df_train[features]
    y_train = df_train[target]
    w_train = df_train['sample_weight'] if 'sample_weight' in df_train.columns else None
    
    X_test = df_test[features]
    y_test = df_test[target]
    w_test = df_test['sample_weight'] if 'sample_weight' in df_test.columns else None
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    print("4. Training LightGBM Model...")
    trainer = RankingTrainer(learning_rate=config.RANKING_LEARNING_RATE, num_leaves=config.RANKING_NUM_LEAVES)
    
    preds = trainer.train(
        X_train, y_train, 
        X_test, y_test, 
        num_boost_round=config.RANKING_NUM_BOOST_ROUND,
        train_weight=w_train,
        test_weight=w_test
    )
    
    print("5. Evaluating Model...")
    query_groups = df_test['user_id'].values
    trainer.evaluate(y_test.values, preds, query_groups=query_groups, k=10)
    
    print("6. Saving Model...")
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    trainer.save_model(config.RANKER_MODEL_PATH)
    print("Ranking Stage Finished Successfully!")

if __name__ == "__main__":
    run_ranking_training()
