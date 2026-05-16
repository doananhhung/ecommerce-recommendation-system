import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from metrics import ndcg_at_k, mrr
import os

class RankingTrainer:
    """
    Huấn luyện mô hình Xếp hạng (Ranking) sử dụng LightGBM.
    """
    def __init__(self, learning_rate: float = 0.05, num_leaves: int = 31):
        self.params = {
            'objective': 'binary',
            'metric': ['binary_logloss', 'auc'],
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'verbose': -1,
            # 'device': 'gpu' # Uncomment if compiled with GPU support
        }
        self.model = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, num_boost_round: int = 100):
        """
        Thực hiện quá trình huấn luyện LightGBM.
        """
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        print("Starting LightGBM training...")
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[test_data]
        )
        
        # Đánh giá cơ bản
        preds = self.model.predict(X_test)
        auc = roc_auc_score(y_test, preds)
        print(f"Validation AUC: {auc:.4f}")
        
        return preds
        
    def evaluate(self, y_true: pd.Series, y_score: np.ndarray, k: int = 10):
        """
        Đánh giá chuyên sâu bằng các độ đo Ranking (NDCG, MRR).
        """
        ndcg = ndcg_at_k(y_true, y_score, k=k)
        model_mrr = mrr(y_true, y_score)
        
        print(f"Evaluation Metrics:")
        print(f"NDCG@{k}: {ndcg:.4f}")
        print(f"MRR:      {model_mrr:.4f}")
        
    def save_model(self, save_path: str):
        """
        Lưu mô hình ra đĩa.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save_model(save_path)
        print(f"Model saved to {save_path}")
