import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    """
    Mô hình Matrix Factorization cơ bản dùng cho bước Recall.
    Sử dụng Embedding để biểu diễn User và Item trong không gian chiều thấp.
    """
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        """
        Args:
            num_users: Tổng số lượng người dùng độc nhất.
            num_items: Tổng số lượng sản phẩm độc nhất.
            embedding_dim: Số chiều của vector nhúng (Embedding Dimension).
        """
        super(MatrixFactorization, self).__init__()
        
        # Lớp Embedding cho User và Item
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Khởi tạo trọng số bằng Xavier Uniform để hỗ trợ hội tụ nhanh hơn
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
    def forward(self, user_idx, item_idx):
        """
        Tính toán dự đoán cho một cặp User-Item.
        """
        # Lấy vector nhúng tương ứng
        u_emb = self.user_embedding(user_idx)
        i_emb = self.item_embedding(item_idx)
        
        # Tính Tích vô hướng (Dot Product)
        dot_product = (u_emb * i_emb).sum(dim=1)
        
        # Vì nhãn là 0 và 1, sử dụng Sigmoid để ép giá trị về khoảng (0, 1)
        return torch.sigmoid(dot_product)
    
    def get_item_embeddings(self):
        """
        Trích xuất toàn bộ vector nhúng của Item để đẩy vào FAISS.
        """
        return self.item_embedding.weight.detach().cpu().numpy()
