import torch
from torch.utils.data import Dataset
import pandas as pd

class ImplicitFeedbackDataset(Dataset):
    """
    Dataset cho dữ liệu phản hồi ẩn (Implicit Feedback) trong hệ thống gợi ý.
    """
    def __init__(self, users: pd.Series, items: pd.Series, labels: pd.Series):
        """
        Khởi tạo dataset.
        
        Args:
            users: Series chứa ID hoặc Index của người dùng.
            items: Series chứa ID hoặc Index của sản phẩm.
            labels: Series chứa nhãn (0 cho negative, 1 cho positive).
        """
        # Chuyển đổi sang tensor, đảm bảo kiểu dữ liệu phù hợp
        self.users = torch.tensor(users.values, dtype=torch.long)
        self.items = torch.tensor(items.values, dtype=torch.long)
        self.labels = torch.tensor(labels.values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]
