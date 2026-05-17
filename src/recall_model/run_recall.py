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

def run_recall_training():
    print("1. Loading Labeled Data...")
    df = pd.read_parquet(config.LABELED_SESSIONS_PATH)
    
    ratio = config.RECALL_NEG_SAMPLE_RATIO
    print(f"1.5. Negative Sampling (1 Positive : {ratio} Negatives)...")
    pos_df = df[df['label'] > 0]
    neg_df = df[df['label'] == 0]
    
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