import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os

class RecallTrainer:
    """
    Lớp quản lý quá trình huấn luyện mô hình Recall.
    """
    def __init__(self, model: nn.Module, learning_rate: float = 0.01, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        # Sử dụng Binary Cross Entropy Loss cho bài toán dự đoán nhị phân (0/1)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def train(self, dataloader: DataLoader, epochs: int = 5):
        """
        Huấn luyện mô hình.
        """
        print(f"Starting training on {self.device}...")
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            
            for users, items, labels in dataloader:
                # Đẩy dữ liệu vào GPU (nếu có)
                users = users.to(self.device)
                items = items.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(users, items)
                
                # Tính loss
                loss = self.criterion(predictions, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}")
            
    def save_model(self, save_path: str):
        """Lưu trọng số mô hình."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
    def save_item_embeddings(self, save_path: str):
        """Lưu Item Embeddings ra file numpy."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        item_embeddings = self.model.get_item_embeddings()
        np.save(save_path, item_embeddings)
        print(f"Item embeddings saved to {save_path} (Shape: {item_embeddings.shape})")
