# Kế hoạch sửa lỗi Data Leakage cho `train_on_recall.py`

## 1. Mục tiêu
Loại bỏ hoàn toàn lỗi **Data Leakage (Look-ahead bias)** trong file `src/ranking_model/train_on_recall.py`. Quá trình tạo dữ liệu huấn luyện (Train-on-Recall) phải sử dụng **Point-in-time features** (tính năng tại đúng thời điểm xảy ra sự kiện) thay vì dùng Snapshot features hiện tại.

## 2. Phân tích nguyên nhân
Hiện tại, `train_on_recall.py` sử dụng hàm `merge()` cơ bản của Pandas để nối `user_feats_clean` và `item_feats_clean` vào dataset:
```python
df_result = df_result.merge(user_feats_clean, on="user_id", how="left")
df_result = df_result.merge(item_feats_clean, on="product_id", how="left")
```
Hai bảng `user_feats` và `item_feats` này là bản ghi cuối cùng (Snapshot) phản ánh tổng số tương tác của toàn bộ dòng thời gian. Việc nối chúng vào các phiên (session) trong quá khứ làm cho mô hình học được thông tin của tương lai.

## 3. Giải pháp triển khai (Point-in-Time Join)
Chúng ta không thể lấy Point-in-time feature một cách trực tiếp từ `df_interactions` cho các Negative candidates (vì user không tương tác với chúng nên chúng không có dòng log tương ứng trong session đó). 

Tuy nhiên, ta có thể dùng tính năng **`pd.merge_asof`** (As-of Merge) của Pandas. Tính năng này cho phép ghép dữ liệu dựa trên mốc thời gian gần nhất trước đó (`event_time`).

### Các bước thực hiện chi tiết:

**Bước 1: Trích xuất lịch sử User State và Item State từ `df_interactions`**
Do `df_interactions` đã được chạy qua hàm `add_point_in_time_features()` trong pipeline, nó đã chứa sẵn dòng thời gian biến động của feature.
- Tạo `user_state_df`: Gồm các cột `['user_id', 'event_time', 'user_total_interactions', 'user_total_sessions']`. Sắp xếp theo `event_time`.
- Tạo `item_state_df`: Gồm các cột `['product_id', 'event_time', 'item_total_interactions', 'item_unique_users', 'item_avg_price', 'category_code', 'brand', 'item_session_popularity']`. Sắp xếp theo `event_time`.

**Bước 2: Sửa đổi cách generate DataFrame kết quả**
Thay vì gọi `df_result.merge(user_feats_clean, ...)`, ta sắp xếp `df_result` theo `event_time` và thực hiện:
```python
# Sắp xếp để chuẩn bị cho merge_asof
df_result = df_result.sort_values("event_time")
user_state_df = user_state_df.sort_values("event_time")
item_state_df = item_state_df.sort_values("event_time")

# Nối Point-in-time User Features
df_result = pd.merge_asof(
    df_result, 
    user_state_df,
    on="event_time",
    by="user_id",
    direction="backward" # Lấy thông tin ngay trước thời điểm xảy ra sự kiện
)

# Nối Point-in-time Item Features
df_result = pd.merge_asof(
    df_result, 
    item_state_df,
    on="event_time",
    by="product_id",
    direction="backward"
)
```

**Bước 3: Dọn dẹp mã nguồn cũ**
- Xóa bỏ việc truy xuất và tải snapshot features ở đầu hàm `generate_train_on_recall_dataset`.
- Điền giá trị dự phòng (`fillna`) cho các trường hợp merge_asof bị rỗng (xảy ra khi một item chưa từng xuất hiện trên hệ thống trước thời điểm đó).

## 4. Lợi ích
- Phương pháp này sử dụng toàn bộ tính năng gốc được cung cấp sẵn của Pandas (Vectorized). Tốc độ nội suy thời gian của `merge_asof` diễn ra rất nhanh và hiệu quả trong C.
- Đảm bảo tính trung thực tuyệt đối của thuật toán Machine Learning. Xóa bỏ hoàn toàn "khả năng ngoại cảm" của AI.
