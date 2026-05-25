import os
import pandas as pd
import numpy as np
from src.config import config
from src.data_pipeline.sessionizer import create_sessions
from src.data_pipeline.pseudo_label import apply_pseudo_labels
from src.data_pipeline.featurizer import (
    add_point_in_time_features,
    extract_item_features,
    extract_user_features,
)

def run_pipeline(raw_data_path: str, output_dir: str, nrows: int):
    print(f"Reading data from {raw_data_path}...")
    df = pd.read_csv(raw_data_path, nrows=nrows)
    
    print("0.5. Preprocessing (Log Transform Price)...")
    if 'price' in df.columns:
        df['price'] = np.log1p(pd.to_numeric(df['price'], errors='coerce').fillna(0))
    
    print("1. Applying Sessionization...")
    df_sessions = create_sessions(df)
    
    print("2. Extracting Serving Feature Snapshots...")
    user_features = extract_user_features(df_sessions)
    item_features = extract_item_features(df_sessions)
    
    print("3. Adding Point-in-Time Training Features...")
    df_training_features = add_point_in_time_features(df_sessions)

    print("4. Applying Pseudo-Labeling...")
    labeled_data = apply_pseudo_labels(df_training_features)
    
    # Create directories if they don't exist
    os.makedirs(os.path.join(output_dir, 'feature_store'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'sessions'), exist_ok=True)
    
    print("5. Saving outputs to disk as Parquet...")
    # Using Parquet for faster I/O and better compression
    user_features.to_parquet(config.USER_FEATURES_PATH, index=False)
    item_features.to_parquet(config.ITEM_FEATURES_PATH, index=False)
    labeled_data.to_parquet(config.LABELED_SESSIONS_PATH, index=False)
    
    print("Pipeline finished successfully!")

if __name__ == "__main__":
    run_pipeline(config.RAW_DATA_PATH, config.DATA_DIR, nrows=config.PIPELINE_NROWS)