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
            
        # Get the exact feature list and order from the trained LightGBM model
        feature_cols = self.model.feature_name()
        
        X = features_df.copy()
        
        # Ensure all required features exist in X
        for col in feature_cols:
            if col not in X.columns:
                if col in ['category_code', 'brand']:
                    X[col] = '<UNKNOWN>'
                else:
                    X[col] = 0.0
        
        # Prepare categorical features
        if 'category_code' in X.columns:
            X['category_code'] = X['category_code'].fillna('<UNKNOWN>').astype('category')
        if 'brand' in X.columns:
            X['brand'] = X['brand'].fillna('<UNKNOWN>').astype('category')
            
        # Prepare numerical features
        numeric_cols = X.select_dtypes(include=['number']).columns
        X[numeric_cols] = X[numeric_cols].fillna(0)
        
        # Select and order features to match model's expected inputs
        X = X[feature_cols]
        
        start_time = time.time()
        preds = self.model.predict(X)
        elapsed = time.time() - start_time
        
        return preds, elapsed

