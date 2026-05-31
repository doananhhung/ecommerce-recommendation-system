import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from src.config import config
from src.serving.pipeline import RecommendationPipeline
from src.ranking_model.lgbm_train import RankingTrainer

def generate_train_on_recall_dataset(pipeline: RecommendationPipeline, df_interactions: pd.DataFrame):
    print("Generating Train-on-Recall dataset...")
    
    # 1. Trích xuất Point-in-time features từ df_interactions thay vì dùng snapshot
    print("Extracting Point-in-time User and Item States...")
    user_state_df = df_interactions[["user_id", "event_time", "user_total_interactions", "user_total_sessions"]].copy()
    user_state_df = user_state_df.sort_values("event_time").drop_duplicates(["user_id", "event_time"], keep="last")
    
    item_cols = ["product_id", "event_time", "item_total_interactions", "item_unique_users", 
                 "item_avg_price", "category_code", "brand", "item_session_popularity"]
    item_cols = [c for c in item_cols if c in df_interactions.columns]
    item_state_df = df_interactions[item_cols].copy()
    item_state_df = item_state_df.sort_values("event_time").drop_duplicates(["product_id", "event_time"], keep="last")
    
    # Group interactions by user_id and custom_session_id to process session by session
    # Sort event_time to keep strict chronological order for session-level times
    session_groups = df_interactions.groupby(["user_id", "custom_session_id"])
    
    unique_sessions = list(session_groups.groups.keys())
    max_sessions = 20000
    if len(unique_sessions) > max_sessions:
        print(f"Sampling {max_sessions} sessions out of {len(unique_sessions)} to prevent memory explosion and speed up training...")
        np.random.seed(42)
        sampled_keys = [unique_sessions[i] for i in np.random.choice(len(unique_sessions), max_sessions, replace=False)]
    else:
        sampled_keys = unique_sessions
        
    records = []
    total_sessions = len(sampled_keys)
    print(f"Total sessions to process: {total_sessions}")
    
    # Process each session
    for idx, key in enumerate(sampled_keys):
        user_id, session_id = key
        session_df = session_groups.get_group(key)
        if idx % 2000 == 0 and idx > 0:
            print(f"Processed {idx}/{total_sessions} sessions...")
        # Map user_id to user_idx
        user_idx = pipeline._user_id_to_idx(user_id)
        if user_idx is None or user_idx >= pipeline.recall_model.user_embedding.num_embeddings or user_idx < 0:
            continue
            
        # 1. Identify actual positive and negative interactions in this session
        # We map product_id to its label and sample_weight
        actual_interactions = {}
        session_event_times = []
        session_product_ids = []
        
        for _, row in session_df.iterrows():
            pid = int(row["product_id"])
            lbl = int(row["label"])
            weight = float(row["sample_weight"])
            session_product_ids.append(pid)
            session_event_times.append(row["event_time"])
            
            # Keep the strongest interaction if duplicate product in same session
            if pid not in actual_interactions or lbl > actual_interactions[pid]["label"]:
                actual_interactions[pid] = {"label": lbl, "sample_weight": weight}
                
        # Session time reference (minimum time in session to prevent lookahead)
        ref_time = min(session_event_times)
        
        # 2. Simulate Dual-Channel Recall stage exactly like Serving
        # --- Channel 1: Long-term Recall ---
        u_emb = pipeline.get_user_embedding(user_idx)
        top_k_long_term = min(config.RECALL_TOP_K_LONG_TERM, pipeline.faiss_searcher.num_items)
        _, long_term_indices, _ = pipeline.faiss_searcher.search(u_emb, top_k=top_k_long_term)
        
        # --- Channel 2: Session-based Recall ---
        session_indices = []
        if len(session_product_ids) > 0:
            valid_session_item_idxs = []
            for pid in session_product_ids:
                if int(pid) in pipeline.product_id_to_idx:
                    valid_session_item_idxs.append(pipeline.product_id_to_idx[int(pid)])
            
            if len(valid_session_item_idxs) > 0:
                import torch
                with torch.no_grad():
                    item_idxs_tensor = torch.tensor(valid_session_item_idxs, dtype=torch.long)
                    item_embs = pipeline.recall_model.item_embedding(item_idxs_tensor)
                    session_vector = item_embs.mean(dim=0).cpu().numpy()
                
                top_k_session = min(config.RECALL_TOP_K_SESSION, pipeline.faiss_searcher.num_items)
                _, session_indices, _ = pipeline.faiss_searcher.search(session_vector, top_k=top_k_session)
                
        # 3. Merge candidates and identify channels
        long_term_set = set(long_term_indices)
        session_set = set(session_indices)
        recall_candidates = list(long_term_set.union(session_set))
        
        # 4. Build records for all candidates in this session
        session_len = len(session_product_ids)
        
        for candidate_idx in recall_candidates:
            # Map candidate_idx back to product_id
            product_id = int(pipeline.item_idx_to_product_id[candidate_idx])
            
            # Determine label and sample weight
            if product_id in actual_interactions:
                label = actual_interactions[product_id]["label"]
                weight = actual_interactions[product_id]["sample_weight"]
            else:
                # Candidate is a true negative (recalled but not interacted by user)
                label = 0
                weight = 0.1 # Small default weight for negative samples
                
            # Construct candidate features dictionary (minimal attributes for fast generation)
            rec = {
                "user_id": user_id,
                "custom_session_id": session_id,
                "product_id": product_id,
                "event_time": ref_time,
                
                # Dynamic session feature
                "user_session_interaction_count": float(session_len),
                
                # Channel indicator features
                "recalled_by_long_term": 1.0 if candidate_idx in long_term_set else 0.0,
                "recalled_by_session": 1.0 if candidate_idx in session_set else 0.0,
                
                # Targets
                "label": label,
                "sample_weight": weight
            }
            records.append(rec)
            
    df_result = pd.DataFrame(records)
    
    # 5. High-performance Vectorized Merge for Point-in-time features
    print("Performing vectorized point-in-time feature join (merge_asof)...")
    
    # Đảm bảo dataset được sort trước khi dùng merge_asof
    df_result = df_result.sort_values("event_time").reset_index(drop=True)
    
    df_result = pd.merge_asof(
        df_result, 
        user_state_df,
        on="event_time",
        by="user_id",
        direction="backward"
    )
    
    df_result = pd.merge_asof(
        df_result, 
        item_state_df,
        on="event_time",
        by="product_id",
        direction="backward"
    )
    
    print(f"Generated dataset with {len(df_result)} rows.")
    return df_result

def run_train_on_recall():
    print("=== Step 1: Initialize Pipeline ===")
    pipeline = RecommendationPipeline(config.MODELS_DIR, config.FEATURE_STORE_DIR)
    pipeline.initialize()
    
    print("\n=== Step 2: Loading Labeled Interactions ===")
    df_interactions = pd.read_parquet(config.LABELED_SESSIONS_PATH)
    
    # Sort events chronologically
    df_interactions = df_interactions.sort_values("event_time").reset_index(drop=True)
    
    print("\n=== Step 3: Generating Recall-based Training Data ===")
    # Generate train-on-recall dataset using Pipeline logic
    df_recall = generate_train_on_recall_dataset(pipeline, df_interactions)
    
    # 4. Post-processing: Handle missing values and types
    df_recall["category_code"] = df_recall["category_code"].fillna("<UNKNOWN>").astype("category")
    df_recall["brand"] = df_recall["brand"].fillna("<UNKNOWN>").astype("category")
    
    # Fill any remaining NaNs in numerical columns
    numeric_cols = df_recall.select_dtypes(include=["number"]).columns
    df_recall[numeric_cols] = df_recall[numeric_cols].fillna(0)
    
    # 5. Time-based Splitting
    df_recall = df_recall.sort_values("event_time").reset_index(drop=True)
    split_idx = int(len(df_recall) * (1 - config.RANKING_TEST_SIZE))
    df_train = df_recall.iloc[:split_idx]
    df_test = df_recall.iloc[split_idx:]
    
    # Select features dynamically from ProjectConfig
    features = []
    for col in config.NUMERICAL_FEATURES:
        if col in df_recall.columns:
            features.append(col)
    for col in config.CATEGORICAL_FEATURES:
        if col in df_recall.columns:
            features.append(col)
            
    target = 'label'
    
    X_train = df_train[features]
    y_train = df_train[target]
    w_train = df_train['sample_weight']
    
    X_test = df_test[features]
    y_test = df_test[target]
    w_test = df_test['sample_weight']
    
    print(f"\nTrain samples: {len(X_train)} (Positive: {y_train.sum()}), Test samples: {len(X_test)} (Positive: {y_test.sum()})")
    print(f"Features used: {features}")
    
    print("\n=== Step 4: Training LightGBM on Recall Candidates ===")
    trainer = RankingTrainer(learning_rate=config.RANKING_LEARNING_RATE, num_leaves=config.RANKING_NUM_LEAVES)
    
    preds = trainer.train(
        X_train, y_train,
        X_test, y_test,
        num_boost_round=config.RANKING_NUM_BOOST_ROUND,
        train_weight=w_train,
        test_weight=w_test
    )
    
    print("\n=== Step 5: Evaluating Model Quality ===")
    query_groups = df_test['user_id'].values
    trainer.evaluate(y_test.values, preds, query_groups=query_groups, k=10)
    
    print("\n=== Step 6: Saving New Ranker Model ===")
    trainer.save_model(config.RANKER_MODEL_PATH)
    print("Train-on-Recall Pipeline Completed Successfully!")

if __name__ == "__main__":
    start_time = time.time()
    run_train_on_recall()
    print(f"\nTotal script execution time: {time.time() - start_time:.2f}s")
