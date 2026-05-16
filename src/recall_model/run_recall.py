import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from dataset import ImplicitFeedbackDataset
from model import MatrixFactorization
from trainer import RecallTrainer

def run_recall_training(data_path: str, models_dir: str, feature_store_dir: str):
    print("1. Loading Labeled Data...")
    df = pd.read_parquet(data_path)
    
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
    dataloader = DataLoader(dataset, batch_size=4096, shuffle=True, num_workers=0)
    
    print("4. Initializing Model & Trainer...")
    # Khởi tạo mô hình Embedding với 64 chiều
    model = MatrixFactorization(num_users=num_users, num_items=num_items, embedding_dim=64)
    trainer = RecallTrainer(model=model, learning_rate=0.01)
    
    print("5. Starting Training Loop...")
    trainer.train(dataloader=dataloader, epochs=5)
    
    print("6. Saving Artifacts...")
    trainer.save_model(os.path.join(models_dir, 'recall_weights.pth'))
    trainer.save_item_embeddings(os.path.join(feature_store_dir, 'item_embeddings.npy'))
    
    print("Recall Stage Finished Successfully!")

if __name__ == "__main__":
    DATA_PATH = 'data/sessions/labeled_sessions.parquet'
    MODELS_DIR = 'models_store/'
    FEATURE_STORE_DIR = 'data/feature_store/'
    run_recall_training(DATA_PATH, MODELS_DIR, FEATURE_STORE_DIR)
