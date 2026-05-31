import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from src.config import config
from src.recall_model.dataset import ImplicitFeedbackDataset
from src.recall_model.model import MatrixFactorization
from src.recall_model.trainer import RecallTrainer

def evaluate_recall(model, df_raw, user_encoder, item_encoder, k=50):
    print(f"\nEvaluating Recall@{k} (Hit Rate@{k}) on Positive Interactions...")
    model.eval()
    import torch
    import faiss
    
    # Filter positive interactions (label > 0)
    pos_df = df_raw[df_raw['label'] > 0].copy()
    if len(pos_df) == 0:
        print("No positive interactions to evaluate recall.")
        return 0.0
        
    # Ensure IDs are within the trained encoder vocabulary
    user_vocab = set(user_encoder.classes_)
    item_vocab = set(item_encoder.classes_)
    pos_df = pos_df[pos_df['user_id'].isin(user_vocab) & pos_df['product_id'].isin(item_vocab)]
    
    if len(pos_df) < 100:
        print(f"WARNING: Warm-start positive test interactions count is {len(pos_df)} (< 100). Skipping Recall@{k} evaluation.")
        return 0.0
        
    # Sample unique users to keep evaluation fast
    sampled_users = pos_df['user_id'].unique()
    if len(sampled_users) > 500:
        np.random.seed(42)
        sampled_users = np.random.choice(sampled_users, 500, replace=False)
        pos_df = pos_df[pos_df['user_id'].isin(sampled_users)]
        
    user_idxs = user_encoder.transform(pos_df['user_id'])
    item_idxs = item_encoder.transform(pos_df['product_id'])
    
    # Group positive item idxs by user idx
    user_pos_items = {}
    for u_idx, i_idx in zip(user_idxs, item_idxs):
        if u_idx not in user_pos_items:
            user_pos_items[u_idx] = []
        user_pos_items[u_idx].append(i_idx)
        
    # Get all item embeddings
    item_embeddings = model.get_item_embeddings()
    
    # Build a temporary FAISS index for evaluation
    index = faiss.IndexFlatIP(config.RECALL_EMBEDDING_DIM)
    index.add(item_embeddings)
    
    hits = 0
    total_positives = 0
    
    device = next(model.parameters()).device
    
    for u_idx, pos_items in user_pos_items.items():
        with torch.no_grad():
            u_tensor = torch.tensor([u_idx], dtype=torch.long).to(device)
            u_emb = model.user_embedding(u_tensor).cpu().numpy()
            
        # Search Top K in FAISS
        _, candidate_indices = index.search(u_emb, k)
        candidate_indices = candidate_indices[0]
        
        for item in pos_items:
            if item in candidate_indices:
                hits += 1
            total_positives += 1
            
    hit_rate = hits / total_positives if total_positives > 0 else 0.0
    print(f"Recall@{k} (Hit Rate@{k}): {hit_rate:.4f} ({hits}/{total_positives} positive interactions captured)")
    return hit_rate

def run_recall_training():
    print("1. Loading Labeled Data...")
    df_raw = pd.read_parquet(config.LABELED_SESSIONS_PATH)
    
    print("1.5. Performing Temporal Split by Session...")
    # Ensure event_time is datetime
    df_raw['event_time'] = pd.to_datetime(df_raw['event_time'])
    
    # Group by custom_session_id to find session_end_time (max event_time)
    session_times = df_raw.groupby('custom_session_id')['event_time'].max().reset_index()
    session_times = session_times.rename(columns={'event_time': 'session_end_time'})
    session_times = session_times.sort_values('session_end_time').reset_index(drop=True)
    
    # 90/10 split on session count
    num_sessions = len(session_times)
    cutoff_idx = int(num_sessions * 0.9)
    train_sessions = set(session_times.iloc[:cutoff_idx]['custom_session_id'])
    
    train_df = df_raw[df_raw['custom_session_id'].isin(train_sessions)].copy()
    test_df = df_raw[~df_raw['custom_session_id'].isin(train_sessions)].copy()
    
    print(f"Total sessions: {num_sessions}. Train sessions: {cutoff_idx}, Test sessions: {num_sessions - cutoff_idx}")
    print(f"Train events: {len(train_df)}, Test events: {len(test_df)}")
    
    print("2. Fitting Encoders on Train Data Only...")
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    user_encoder.fit(train_df['user_id'])
    item_encoder.fit(train_df['product_id'])
    
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    print(f"Vocabularies fitted - Total Users: {num_users}, Total Items: {num_items}")
    
    print("3. Mixing Negative Samples...")
    # Positive items (label > 0) from train_df. Keep duplicates for frequency/strength signal.
    pos_df = train_df[train_df['label'] > 0][['user_id', 'product_id', 'label']].copy()
    pos_df['label'] = 1.0  # Force to standard 1.0 for model training
    
    # Hard Negatives from train_df (label == 0), unique by (user_id, product_id)
    hard_neg_candidates = train_df[train_df['label'] == 0][['user_id', 'product_id']].drop_duplicates()
    target_hard_negatives = len(pos_df) * config.HARD_NEG_RATIO
    if len(hard_neg_candidates) > target_hard_negatives:
        hard_neg_df = hard_neg_candidates.sample(n=target_hard_negatives, random_state=42).copy()
    else:
        hard_neg_df = hard_neg_candidates.copy()
        print(f"WARNING: Available hard negatives ({len(hard_neg_candidates)}) is less than target ({target_hard_negatives}). Using all.")
    hard_neg_df['label'] = 0.0
    
    # Easy Negatives (Global Random)
    observed_train_pairs = train_df[['user_id', 'product_id']].drop_duplicates()
    easy_neg_list = []
    pool_size = 0
    max_attempts = 100
    attempt = 0
    
    user_pool = user_encoder.classes_
    item_pool = item_encoder.classes_
    target_easy_negatives = len(pos_df) * config.EASY_NEG_RATIO
    
    np.random.seed(42)
    
    while pool_size < target_easy_negatives and attempt < max_attempts:
        attempt += 1
        batch_size = max(int(target_easy_negatives * 1.5), 1000)
        
        random_users = np.random.choice(user_pool, size=batch_size, replace=True)
        random_items = np.random.choice(item_pool, size=batch_size, replace=True)
        
        batch_df = pd.DataFrame({
            'user_id': random_users,
            'product_id': random_items
        })
        
        batch_df = batch_df.drop_duplicates(['user_id', 'product_id'])
        
        # Anti-join via left-join
        merged = pd.merge(
            batch_df,
            observed_train_pairs.assign(is_observed=1),
            on=['user_id', 'product_id'],
            how='left'
        )
        candidates = merged[merged['is_observed'].isna()][['user_id', 'product_id']].copy()
        
        if len(candidates) > 0:
            easy_neg_list.append(candidates)
            temp_pool = pd.concat(easy_neg_list).drop_duplicates(['user_id', 'product_id'])
            pool_size = len(temp_pool)
            
    if len(easy_neg_list) > 0:
        easy_neg_df = pd.concat(easy_neg_list).drop_duplicates(['user_id', 'product_id'])
        if len(easy_neg_df) > target_easy_negatives:
            easy_neg_df = easy_neg_df.sample(n=target_easy_negatives, random_state=42).copy()
    else:
        easy_neg_df = pd.DataFrame(columns=['user_id', 'product_id'])
        
    if len(easy_neg_df) < target_easy_negatives:
        print(f"WARNING: Generated only {len(easy_neg_df)} easy negatives (target: {target_easy_negatives}) after {attempt} attempts.")
    easy_neg_df['label'] = 0.0
    
    # Concatenate all sets
    df = pd.concat([pos_df, hard_neg_df, easy_neg_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"After sampling: {len(pos_df)} positives, {len(hard_neg_df)} hard negatives, {len(easy_neg_df)} easy negatives.")
    
    # Map IDs to Indices
    df['user_idx'] = user_encoder.transform(df['user_id'])
    df['item_idx'] = item_encoder.transform(df['product_id'])
    
    print("4. Preparing DataLoader...")
    dataset = ImplicitFeedbackDataset(df['user_idx'], df['item_idx'], df['label'])
    dataloader = DataLoader(dataset, batch_size=config.RECALL_BATCH_SIZE, shuffle=True, num_workers=0)
    
    print("5. Initializing Model & Trainer...")
    model = MatrixFactorization(num_users=num_users, num_items=num_items, embedding_dim=config.RECALL_EMBEDDING_DIM)
    trainer = RecallTrainer(model=model, learning_rate=config.RECALL_LEARNING_RATE)
    
    print("6. Starting Training Loop...")
    trainer.train(dataloader=dataloader, epochs=config.RECALL_EPOCHS)
    
    # Run recall model quality evaluation (Recall@50 / Hit Rate@50) on test_df
    evaluate_recall(model, test_df, user_encoder, item_encoder, k=50)
    
    print("7. Saving Artifacts...")
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    trainer.save_model(config.RECALL_WEIGHTS_PATH)
    trainer.save_item_embeddings(config.ITEM_EMBEDDINGS_PATH)
    joblib.dump(user_encoder, config.USER_ENCODER_PATH)
    joblib.dump(item_encoder, config.ITEM_ENCODER_PATH)
    
    metadata = {
        'num_users': int(num_users),
        'num_items': int(num_items),
        'embedding_dim': config.RECALL_EMBEDDING_DIM,
    }
    with open(config.RECALL_METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"Encoders saved to {config.MODELS_DIR}")
    
    print("Recall Stage Finished Successfully!")

if __name__ == "__main__":
    run_recall_training()