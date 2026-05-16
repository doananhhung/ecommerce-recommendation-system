import pandas as pd

def apply_pseudo_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo nhãn giả (pseudo-labels) dựa trên các sự kiện trong một phiên của người dùng.
    0 = Negative (Chỉ xem)
    1 = Positive (Đã thêm vào giỏ hàng hoặc mua)
    
    Args:
        df: DataFrame chứa các sự kiện đã được gom nhóm (có 'custom_session_id', 'product_id', 'user_id', 'event_type').
        
    Returns:
        DataFrame chứa các mẫu đã được gán nhãn (label).
    """
    df = df.copy()
    
    # Gán trọng số cho từng loại sự kiện
    event_weights = {'view': 0, 'cart': 1, 'purchase': 1}
    df['event_weight'] = df['event_type'].map(event_weights)
    
    # Lấy hành động có giá trị cao nhất trong phiên cho mỗi sản phẩm của từng user
    labeled_df = df.groupby(['custom_session_id', 'product_id', 'user_id'])['event_weight'].max().reset_index()
    labeled_df.rename(columns={'event_weight': 'label'}, inplace=True)
    
    return labeled_df
