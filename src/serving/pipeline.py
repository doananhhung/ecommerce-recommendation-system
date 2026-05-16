import os
import torch
import numpy as np
import pandas as pd
import time
from typing import List, Dict, Any

from .faiss_index import FaissSearcher
from .ranker import LightGBMRanker

# Note: We need a simplified way to get User Embedding. We'll load the PyTorch weights.
# We also need a way to look up features. Since this is an offline serving setup,
# we'll load the parquet files into pandas DataFrames acting as a simple in-memory Feature Store.

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from recall_model.model import MatrixFactorization

class RecommendationPipeline:
    def __init__(self, models_dir: str, feature_store_dir: str):
        self.models_dir = models_dir
        self.feature_store_dir = feature_store_dir
        
        self.faiss_searcher = None
        self.ranker = None
        self.recall_model = None
        
        # Simple In-Memory Feature Store
        self.user_features = None
        self.item_features = None
        
    def initialize(self):
        print("Initializing Recommendation Pipeline...")
        start_time = time.time()
        
        # 1. Load Feature Stores
        print("Loading Feature Stores into memory...")
        self.user_features = pd.read_parquet(os.path.join(self.feature_store_dir, 'user_features.parquet'))
        # Create an index for faster lookup
        self.user_features.set_index('user_id', inplace=True)
        
        self.item_features = pd.read_parquet(os.path.join(self.feature_store_dir, 'item_features.parquet'))
        
        # 2. Init FAISS Searcher
        self.faiss_searcher = FaissSearcher(os.path.join(self.feature_store_dir, 'item_embeddings.npy'))
        self.faiss_searcher.build_index()
        
        # 3. Init Recall Model for User Embeddings
        # We need num_users, num_items to init model. We can infer it from the embeddings shape and user features length
        num_items = self.faiss_searcher.num_items
        num_users = len(self.user_features)
        
        print("Loading PyTorch Recall Model...")
        self.recall_model = MatrixFactorization(num_users=num_users, num_items=num_items, embedding_dim=64)
        weights_path = os.path.join(self.models_dir, 'recall_weights.pth')
        # Map location cpu since we do serving on CPU for simplicity
        self.recall_model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True))
        self.recall_model.eval()
        
        # 4. Init LightGBM Ranker
        self.ranker = LightGBMRanker(os.path.join(self.models_dir, 'ranker_model.txt'))
        self.ranker.load_model()
        
        elapsed = time.time() - start_time
        print(f"Pipeline initialized in {elapsed:.3f}s")
        
    def get_user_embedding(self, user_idx: int) -> np.ndarray:
        with torch.no_grad():
            user_tensor = torch.tensor([user_idx], dtype=torch.long)
            emb = self.recall_model.user_embedding(user_tensor)
            return emb.numpy()[0]
            
    def recommend(self, user_idx: int, top_n: int = 20) -> Dict[str, Any]:
        """
        Thực hiện toàn bộ luồng gợi ý: Recall -> Feature Lookup -> Rank.
        (Giả định nhận user_idx đã được encode).
        """
        metrics = {}
        total_start = time.time()
        
        # 1. Check Cold-Start
        if user_idx >= len(self.user_features) or user_idx < 0:
            return {
                "error": "Cold-start user (or out of bounds index). Fallback logic not fully implemented.",
                "recommendations": []
            }
            
        # 2. Lấy User Embedding
        u_start = time.time()
        u_emb = self.get_user_embedding(user_idx)
        metrics['user_emb_time_ms'] = (time.time() - u_start) * 1000
        
        # 3. Recall (FAISS)
        _, candidate_indices, faiss_time = self.faiss_searcher.search(u_emb, top_k=200)
        metrics['faiss_search_time_ms'] = faiss_time * 1000
        
        # 4. Feature Lookup & Join
        lookup_start = time.time()
        # Trích xuất user feature (dạng dict/Series)
        try:
            # We use index in pandas. In our simplified implementation, we assume user_idx == index position.
            # But the dataset has original user_id. We'd ideally need a mapper.
            # For this prototype, we'll try to get it by positional iloc since user_features might be sorted, 
            # or just take the first matching row if we assumed user_idx maps linearly.
            # Warning: Real systems need a Key-Value lookup (Redis).
            # To be safe, we just use a generic user profile if lookup fails.
            u_feats = self.user_features.iloc[user_idx].to_dict() 
        except IndexError:
            u_feats = {'user_total_interactions': 0, 'user_total_sessions': 0}
            
        # Trích xuất item features cho 200 candidates
        # Tương tự, ta dùng iloc giả định item_idx == row number.
        candidates_df = self.item_features.iloc[candidate_indices].copy()
        
        # Gán thông tin user vào DataFrame ứng viên
        for k, v in u_feats.items():
            candidates_df[k] = v
            
        metrics['feature_lookup_time_ms'] = (time.time() - lookup_start) * 1000
        
        # 5. Ranking (LightGBM)
        preds, rank_time = self.ranker.predict(candidates_df)
        metrics['ranking_time_ms'] = rank_time * 1000
        
        # 6. Lọc Top N
        sort_start = time.time()
        # Sắp xếp index của preds
        top_n_local_indices = np.argsort(preds)[::-1][:top_n]
        top_n_global_indices = candidate_indices[top_n_local_indices]
        top_n_scores = preds[top_n_local_indices]
        
        metrics['sort_time_ms'] = (time.time() - sort_start) * 1000
        metrics['total_latency_ms'] = (time.time() - total_start) * 1000
        
        recommendations = [
            {"item_idx": int(idx), "score": float(score)} 
            for idx, score in zip(top_n_global_indices, top_n_scores)
        ]
        
        return {
            "user_idx": user_idx,
            "recommendations": recommendations,
            "latency_metrics": metrics
        }
