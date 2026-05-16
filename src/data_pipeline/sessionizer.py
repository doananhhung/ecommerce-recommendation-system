import pandas as pd

def create_sessions(df: pd.DataFrame, threshold_minutes: int = 30) -> pd.DataFrame:
    """
    Gom nhóm các hành vi của người dùng thành các phiên (session) theo thời gian.
    
    Args:
        df: DataFrame chứa lịch sử tương tác có cột 'user_id' và 'event_time'.
        threshold_minutes: Khoảng thời gian tối đa giữa 2 sự kiện để được tính là cùng 1 phiên.
        
    Returns:
        DataFrame đã được thêm cột 'custom_session_id'.
    """
    df = df.copy()
    
    # Đảm bảo event_time là datetime
    if not pd.api.types.is_datetime64_any_dtype(df['event_time']):
        df['event_time'] = pd.to_datetime(df['event_time'])
        
    # Sắp xếp dữ liệu theo user_id và event_time
    df = df.sort_values(by=['user_id', 'event_time']).reset_index(drop=True)
    
    # Tính khoảng thời gian chênh lệch giữa các sự kiện liên tiếp của cùng 1 user
    df['time_diff'] = df.groupby('user_id')['event_time'].diff()
    
    # Đánh dấu sự kiện đầu tiên của 1 session mới
    is_new_session = df['time_diff'].dt.total_seconds() > (threshold_minutes * 60)
    is_new_session = is_new_session | df['time_diff'].isna()
    
    # Tạo session_id bằng cách cộng dồn (cumsum)
    df['custom_session_id'] = is_new_session.cumsum()
    
    return df.drop(columns=['time_diff'])
