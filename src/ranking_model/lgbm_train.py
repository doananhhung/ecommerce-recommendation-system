import os
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from src.ranking_model.metrics import mrr, ndcg_at_k
except ModuleNotFoundError:
    from metrics import mrr, ndcg_at_k


class RankingTrainer:
    """
    Train a LightGBM binary ranker/scorer.
    """

    def __init__(self, learning_rate: float = 0.05, num_leaves: int = 31):
        self.params = {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "verbose": -1,
        }
        self.model = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        num_boost_round: int = 100,
        train_weight: Optional[pd.Series] = None,
        test_weight: Optional[pd.Series] = None,
    ):
        train_data = lgb.Dataset(X_train, label=y_train, weight=train_weight)
        test_data = lgb.Dataset(X_test, label=y_test, weight=test_weight, reference=train_data)

        print("Starting LightGBM training...")
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[test_data],
        )

        preds = self.model.predict(X_test)
        auc = roc_auc_score(y_test, preds, sample_weight=test_weight)
        print(f"Validation AUC: {auc:.4f}")

        return preds

    def evaluate(self, y_true: pd.Series, y_score: np.ndarray, k: int = 10):
        ndcg = ndcg_at_k(y_true, y_score, k=k)
        model_mrr = mrr(y_true, y_score)

        print("Evaluation Metrics:")
        print(f"NDCG@{k}: {ndcg:.4f}")
        print(f"MRR:      {model_mrr:.4f}")

    def save_model(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save_model(save_path)
        print(f"Model saved to {save_path}")
