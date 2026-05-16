import os
import pandas as pd
from sessionizer import create_sessions
from pseudo_label import apply_pseudo_labels
from featurizer import extract_user_features, extract_item_features

def run_pipeline(raw_data_path: str, output_dir: str, nrows: int):
    print(f"Reading data from {raw_data_path}...")
    df = pd.read_csv(raw_data_path, nrows=nrows)
    
    print("1. Applying Sessionization...")
    df_sessions = create_sessions(df)
    
    print("2. Extracting Features...")
    user_features = extract_user_features(df_sessions)
    item_features = extract_item_features(df_sessions)
    
    print("3. Applying Pseudo-Labeling...")
    labeled_data = apply_pseudo_labels(df_sessions)
    
    # Create directories if they don't exist
    os.makedirs(os.path.join(output_dir, 'feature_store'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'sessions'), exist_ok=True)
    
    print("4. Saving outputs to disk as Parquet...")
    # Using Parquet for faster I/O and better compression
    user_features.to_parquet(os.path.join(output_dir, 'feature_store', 'user_features.parquet'), index=False)
    item_features.to_parquet(os.path.join(output_dir, 'feature_store', 'item_features.parquet'), index=False)
    labeled_data.to_parquet(os.path.join(output_dir, 'sessions', 'labeled_sessions.parquet'), index=False)
    
    print("Pipeline finished successfully!")

if __name__ == "__main__":
    # Thay đổi nrows nếu muốn chạy trên toàn bộ tập dữ liệu (bỏ nrows hoặc set None)
    # Tạm thời để 1,000,000 dòng để tránh OOM trong lúc thử nghiệm
    DATA_PATH = 'data/raw/2019-Oct.csv'
    OUTPUT_DIR = 'data/'
    run_pipeline(DATA_PATH, OUTPUT_DIR, nrows=1000000)
