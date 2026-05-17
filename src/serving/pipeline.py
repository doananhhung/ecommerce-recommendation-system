import os
import time
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import torch

from src.recall_model.model import MatrixFactorization
from src.serving.faiss_index import FaissSearcher
from src.serving.ranker import LightGBMRanker


class RecommendationPipeline:
    def __init__(self, models_dir: str, feature_store_dir: str):
        self.models_dir = models_dir
        self.feature_store_dir = feature_store_dir

        self.faiss_searcher = None
        self.ranker = None
        self.recall_model = None

        self.user_features = None
        self.item_features = None
        self.user_id_to_idx: Dict[int, int] = {}
        self.idx_to_user_id = None
        self.item_idx_to_product_id = None

    def initialize(self):
        print("Initializing Recommendation Pipeline...")
        start_time = time.time()

        print("Loading Feature Stores into memory...")
        self.user_features = pd.read_parquet(
            os.path.join(self.feature_store_dir, "user_features.parquet")
        )
        self.item_features = pd.read_parquet(
            os.path.join(self.feature_store_dir, "item_features.parquet")
        )
        self._load_id_mappings()

        self.user_features.set_index("user_id", inplace=True, drop=False)
        self.item_features.set_index("product_id", inplace=True, drop=False)

        self.faiss_searcher = FaissSearcher(
            os.path.join(self.feature_store_dir, "item_embeddings.npy")
        )
        self.faiss_searcher.build_index()

        print("Loading PyTorch Recall Model...")
        weights_path = os.path.join(self.models_dir, "recall_weights.pth")
        state_dict = torch.load(
            weights_path,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        num_users, embedding_dim = state_dict["user_embedding.weight"].shape
        num_items = state_dict["item_embedding.weight"].shape[0]
        self._validate_artifacts(num_users=num_users, num_items=num_items)

        self.recall_model = MatrixFactorization(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
        )
        self.recall_model.load_state_dict(state_dict)
        self.recall_model.eval()

        self.ranker = LightGBMRanker(os.path.join(self.models_dir, "ranker_model.txt"))
        self.ranker.load_model()

        elapsed = time.time() - start_time
        print(f"Pipeline initialized in {elapsed:.3f}s")

    def _load_id_mappings(self):
        user_encoder_path = os.path.join(self.models_dir, "user_encoder.joblib")
        item_encoder_path = os.path.join(self.models_dir, "item_encoder.joblib")

        if os.path.exists(user_encoder_path) and os.path.exists(item_encoder_path):
            print("Loading ID encoders from model artifacts...")
            user_classes = np.asarray(joblib.load(user_encoder_path).classes_)
            item_classes = np.asarray(joblib.load(item_encoder_path).classes_)
        else:
            print("Encoder artifacts not found. Rebuilding mappings from feature store.")
            user_classes = np.sort(self.user_features["user_id"].unique())
            item_classes = np.sort(self.item_features["product_id"].unique())

        self.idx_to_user_id = user_classes
        self.item_idx_to_product_id = item_classes
        self.user_id_to_idx = {
            int(user_id): int(user_idx)
            for user_idx, user_id in enumerate(user_classes)
        }

    def _validate_artifacts(self, num_users: int, num_items: int):
        if len(self.idx_to_user_id) < num_users:
            raise ValueError(
                f"User mapping has {len(self.idx_to_user_id)} users, "
                f"but recall model expects {num_users}."
            )
        if len(self.item_idx_to_product_id) != num_items:
            raise ValueError(
                f"Item mapping has {len(self.item_idx_to_product_id)} items, "
                f"but recall model expects {num_items}."
            )
        if self.faiss_searcher.num_items != num_items:
            raise ValueError(
                f"FAISS index has {self.faiss_searcher.num_items} items, "
                f"but recall model expects {num_items}."
            )

    def get_user_embedding(self, user_idx: int) -> np.ndarray:
        with torch.no_grad():
            user_tensor = torch.tensor([user_idx], dtype=torch.long)
            emb = self.recall_model.user_embedding(user_tensor)
            return emb.numpy()[0]

    def recommend(self, user_id: int, top_n: int = 20) -> Dict[str, Any]:
        user_idx = self._user_id_to_idx(user_id)
        if user_idx is None:
            return {
                "error": "Cold-start user. Fallback logic not fully implemented.",
                "recommendations": [],
            }

        return self.recommend_by_index(user_idx=user_idx, top_n=top_n, user_id=user_id)

    def recommend_by_index(
        self,
        user_idx: int,
        top_n: int = 20,
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        metrics = {}
        total_start = time.time()

        if user_idx >= self.recall_model.user_embedding.num_embeddings or user_idx < 0:
            return {
                "error": "Cold-start user (or out of bounds index). Fallback logic not fully implemented.",
                "recommendations": [],
            }
        if user_id is None:
            user_id = int(self.idx_to_user_id[user_idx])

        u_start = time.time()
        u_emb = self.get_user_embedding(user_idx)
        metrics["user_emb_time_ms"] = (time.time() - u_start) * 1000

        top_k = min(200, self.faiss_searcher.num_items)
        _, candidate_indices, faiss_time = self.faiss_searcher.search(u_emb, top_k=top_k)
        metrics["faiss_search_time_ms"] = faiss_time * 1000

        lookup_start = time.time()
        if user_id in self.user_features.index:
            u_feats = self.user_features.loc[user_id].to_dict()
        else:
            u_feats = {"user_total_interactions": 0, "user_total_sessions": 0}

        product_ids = self.item_idx_to_product_id[candidate_indices]
        candidates_df = self.item_features.reindex(product_ids).copy()
        candidates_df["product_id"] = product_ids
        candidates_df["item_idx"] = candidate_indices

        for key, value in u_feats.items():
            candidates_df[key] = value

        metrics["feature_lookup_time_ms"] = (time.time() - lookup_start) * 1000

        preds, rank_time = self.ranker.predict(candidates_df)
        metrics["ranking_time_ms"] = rank_time * 1000

        sort_start = time.time()
        top_n_local_indices = np.argsort(preds)[::-1][:top_n]
        top_n_global_indices = candidate_indices[top_n_local_indices]
        top_n_product_ids = product_ids[top_n_local_indices]
        top_n_scores = preds[top_n_local_indices]

        metrics["sort_time_ms"] = (time.time() - sort_start) * 1000
        metrics["total_latency_ms"] = (time.time() - total_start) * 1000

        recommendations = [
            {
                "product_id": int(product_id),
                "item_idx": int(item_idx),
                "score": float(score),
            }
            for product_id, item_idx, score in zip(
                top_n_product_ids,
                top_n_global_indices,
                top_n_scores,
            )
        ]

        return {
            "user_id": int(user_id),
            "user_idx": int(user_idx),
            "recommendations": recommendations,
            "latency_metrics": metrics,
        }

    def _user_id_to_idx(self, user_id: int) -> Optional[int]:
        return self.user_id_to_idx.get(int(user_id))
