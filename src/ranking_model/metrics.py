import numpy as np

def ndcg_at_k(y_true, y_score, k=10):
    """
    Tính Normalized Discounted Cumulative Gain tại mức K (NDCG@K).
    
    Args:
        y_true: Mảng các nhãn thực tế (1: positive, 0: negative)
        y_score: Mảng điểm số dự đoán từ mô hình (xác suất CTR)
        k: Số lượng item cần lấy ở Top K
        
    Returns:
        float: Giá trị NDCG@K
    """
    # Xếp hạng giảm dần dựa trên điểm số (y_score)
    order = np.argsort(y_score)[::-1][:k]
    y_true_sorted = np.array(y_true)[order]
    
    # Tính Discounted Cumulative Gain (DCG)
    dcg = np.sum((2**y_true_sorted - 1) / np.log2(np.arange(2, len(y_true_sorted) + 2)))
    
    # Tính Ideal DCG (IDCG) - xếp hạng tốt nhất có thể
    ideal_order = np.argsort(y_true)[::-1][:k]
    ideal_y_true_sorted = np.array(y_true)[ideal_order]
    idcg = np.sum((2**ideal_y_true_sorted - 1) / np.log2(np.arange(2, len(ideal_y_true_sorted) + 2)))
    
    if idcg == 0:
        return 0.0
    return dcg / idcg

def mrr(y_true, y_score):
    """
    Tính Mean Reciprocal Rank (MRR).
    
    Args:
        y_true: Mảng các nhãn thực tế
        y_score: Mảng điểm số dự đoán
        
    Returns:
        float: Giá trị MRR
    """
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.array(y_true)[order]
    
    # Tìm index của positive item đầu tiên (có label > 0)
    positive_indices = np.where(y_true_sorted > 0)[0]
    
    if len(positive_indices) == 0:
        return 0.0
        
    # Rank bắt đầu từ 1, do đó +1
    first_positive_rank = positive_indices[0] + 1
    return 1.0 / first_positive_rank
