import pandas as pd

def extract_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trích xuất các đặc trưng tĩnh của người dùng từ lịch sử tương tác.
    """
    # Tính tổng số lượt tương tác của mỗi user
    user_features = df.groupby('user_id').size().reset_index(name='user_total_interactions')
    
    # Tính số lượng phiên duy nhất của mỗi user
    if 'custom_session_id' in df.columns:
        user_sessions = df.groupby('user_id')['custom_session_id'].nunique().reset_index(name='user_total_sessions')
        user_features = user_features.merge(user_sessions, on='user_id', how='left')
        
    return user_features

def extract_item_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trích xuất các đặc trưng tĩnh của sản phẩm từ lịch sử tương tác.
    """
    # Tính tổng số lượt tương tác của mỗi item
    item_features = df.groupby('product_id').size().reset_index(name='item_total_interactions')
    
    # Tính số lượng user duy nhất đã tương tác với item
    item_users = df.groupby('product_id')['user_id'].nunique().reset_index(name='item_unique_users')
    item_features = item_features.merge(item_users, on='product_id', how='left')
    
    # Lấy giá trị trung bình của giá sản phẩm (nếu có trường price)
    if 'price' in df.columns:
        item_price = df.groupby('product_id')['price'].mean().reset_index(name='item_avg_price')
        item_features = item_features.merge(item_price, on='product_id', how='left')
        
    return item_features
