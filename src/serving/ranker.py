import lightgbm as lgb
import pandas as pd
import numpy as np
import time
from typing import Tuple, Any

class LightGBMRanker:
    """
    Quản lý việc tải mô hình LightGBM và dự đoán xếp hạng (Ranking).
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        
    def load_model(self):
        """
        Tải mô hình từ đĩa.
        """
        print(f"Loading LightGBM ranker from {self.model_path}...")
        start_time = time.time()
        self.model = lgb.Booster(model_file=self.model_path)
        elapsed = time.time() - start_time
        print(f"LightGBM model loaded in {elapsed:.3f}s")
        
    def predict(self, features_df: pd.DataFrame) -> Tuple[Any, float]:
        """
        Dự đoán điểm số (CTR) cho các ứng viên.
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Call load_model() first.")
            
        # LightGBM requires specific feature order, which was used during training
        # 'user_total_interactions', 'user_total_sessions', 'item_total_interactions', 'item_unique_users', 'item_avg_price'
        feature_cols = [
            'user_total_interactions', 'user_total_sessions', 
            'item_total_interactions', 'item_unique_users', 'item_avg_price'
        ]
        
        # Ensure columns exist and fillna
        X = features_df[feature_cols].copy()
        X.fillna(0, inplace=True)
        
        start_time = time.time()
        preds = self.model.predict(X)
        elapsed = time.time() - start_time
        
        return preds, elapsed
