import numpy as np
import pandas as pd

def _ndcg_at_k_single(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1][:k]
    y_true_sorted = np.array(y_true)[order]
    
    dcg = np.sum((2**y_true_sorted - 1) / np.log2(np.arange(2, len(y_true_sorted) + 2)))
    
    ideal_order = np.argsort(y_true)[::-1][:k]
    ideal_y_true_sorted = np.array(y_true)[ideal_order]
    idcg = np.sum((2**ideal_y_true_sorted - 1) / np.log2(np.arange(2, len(ideal_y_true_sorted) + 2)))
    
    if idcg == 0:
        return 0.0
    return dcg / idcg

def _mrr_single(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.array(y_true)[order]
    
    positive_indices = np.where(y_true_sorted > 0)[0]
    
    if len(positive_indices) == 0:
        return 0.0
        
    first_positive_rank = positive_indices[0] + 1
    return 1.0 / first_positive_rank

def ndcg_at_k(y_true, y_score, query_groups=None, k=10):
    """
    Tính Normalized Discounted Cumulative Gain tại mức K (NDCG@K).
    Nếu query_groups được cung cấp, tính NDCG cho từng group rồi lấy trung bình.
    """
    if query_groups is None:
        return _ndcg_at_k_single(y_true, y_score, k)
        
    df = pd.DataFrame({'y_true': y_true, 'y_score': y_score, 'group': query_groups})
    scores = df.groupby('group').apply(lambda x: _ndcg_at_k_single(x['y_true'].values, x['y_score'].values, k))
    return scores.mean()

def mrr(y_true, y_score, query_groups=None):
    """
    Tính Mean Reciprocal Rank (MRR).
    Nếu query_groups được cung cấp, tính MRR cho từng group rồi lấy trung bình.
    """
    if query_groups is None:
        return _mrr_single(y_true, y_score)
        
    df = pd.DataFrame({'y_true': y_true, 'y_score': y_score, 'group': query_groups})
    scores = df.groupby('group').apply(lambda x: _mrr_single(x['y_true'].values, x['y_score'].values))
    return scores.mean()
