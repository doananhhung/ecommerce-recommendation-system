import faiss
import numpy as np
import os
import time

class FaissSearcher:
    """
    Quản lý việc tải vector sản phẩm và tìm kiếm KNN (K-Nearest Neighbors) bằng FAISS.
    """
    def __init__(self, embeddings_path: str):
        self.embeddings_path = embeddings_path
        self.index = None
        self.num_items = 0
        self.dim = 0
        
    def build_index(self):
        """
        Tải embeddings từ đĩa và xây dựng FAISS Index.
        Sử dụng IndexFlatIP (Inner Product) vì trong thuật toán Recall đã huấn luyện bằng Dot Product.
        """
        print(f"Loading item embeddings from {self.embeddings_path}...")
        start_time = time.time()
        
        item_embeddings = np.load(self.embeddings_path)
        self.num_items, self.dim = item_embeddings.shape
        
        # Build index
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(item_embeddings)
        
        elapsed = time.time() - start_time
        print(f"FAISS Index built with {self.num_items} items (Dim: {self.dim}) in {elapsed:.3f}s")
        
    def search(self, user_vector: np.ndarray, top_k: int = 200):
        """
        Tìm kiếm Top K sản phẩm gần nhất với user_vector.
        
        Args:
            user_vector: Vector nhúng của user (shape: [1, dim]).
            top_k: Số lượng sản phẩm trả về.
            
        Returns:
            scores: Khoảng cách (điểm số) của Top K items.
            indices: Index của Top K items.
        """
        if self.index is None:
            raise ValueError("FAISS Index is not built yet. Call build_index() first.")
            
        # FAISS yêu cầu input là float32, dạng ma trận 2D
        user_vector = user_vector.astype(np.float32)
        if len(user_vector.shape) == 1:
            user_vector = np.expand_dims(user_vector, axis=0)
            
        # Tìm kiếm
        start_time = time.time()
        scores, indices = self.index.search(user_vector, top_k)
        elapsed = time.time() - start_time
        
        # Chỉ trả về kết quả cho 1 user
        return scores[0], indices[0], elapsed
