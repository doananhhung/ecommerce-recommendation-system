import os
import json
import joblib
import pandas as pd
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
    import numpy as np
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
    
    if len(pos_df) == 0:
        print("No valid positive interactions for evaluation within encoder vocabulary.")
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
    
    ratio = config.RECALL_NEG_SAMPLE_RATIO
    print(f"1.5. Negative Sampling (1 Positive : {ratio} Negatives)...")
    pos_df = df_raw[df_raw['label'] > 0]
    neg_df = df_raw[df_raw['label'] == 0]
    
    num_negatives = len(pos_df) * ratio
    if len(neg_df) > num_negatives:
        neg_df = neg_df.sample(n=num_negatives, random_state=42)
        
    df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"After sampling: {len(pos_df)} positives, {len(neg_df)} negatives.")
    
    print("2. Label Encoding (IDs -> Indices)...")
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    df['user_idx'] = user_encoder.fit_transform(df['user_id'])
    df['item_idx'] = item_encoder.fit_transform(df['product_id'])
    
    num_users = df['user_idx'].nunique()
    num_items = df['item_idx'].nunique()
    print(f"Total Users: {num_users}, Total Items: {num_items}")
    
    print("3. Preparing DataLoader...")
    dataset = ImplicitFeedbackDataset(df['user_idx'], df['item_idx'], df['label'])
    dataloader = DataLoader(dataset, batch_size=config.RECALL_BATCH_SIZE, shuffle=True, num_workers=0)
    
    print("4. Initializing Model & Trainer...")
    model = MatrixFactorization(num_users=num_users, num_items=num_items, embedding_dim=config.RECALL_EMBEDDING_DIM)
    trainer = RecallTrainer(model=model, learning_rate=config.RECALL_LEARNING_RATE)
    
    print("5. Starting Training Loop...")
    trainer.train(dataloader=dataloader, epochs=config.RECALL_EPOCHS)
    
    # Run recall model quality evaluation (Recall@50 / Hit Rate@50)
    evaluate_recall(model, df_raw, user_encoder, item_encoder, k=50)
    
    print("6. Saving Artifacts...")
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