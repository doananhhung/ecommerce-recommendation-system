# Kế hoạch sửa lỗi Sample Selection Bias cho Recall Model (Final v6 - Sẵn sàng Code)

Bản thiết kế đã hoàn tất mọi vòng review. Không còn blocker thuật toán. Mọi quyết định đều được chốt rõ ràng, sẵn sàng chuyển hóa thành code.

## 1. Mục tiêu
Áp dụng **Global Random Negative Sampling** vào mô hình Recall, vá Temporal Leakage, quản trị Cold-start, chuẩn hóa Evaluation và tối ưu hiệu suất.

## 2. Giải pháp triển khai chi tiết

### Bước 1: Temporal Split nguyên vẹn theo Session
- Gom nhóm dữ liệu theo `custom_session_id`, tìm ra `session_end_time`.
- Sắp xếp các session theo `session_end_time`. Lấy mốc Cutoff 90% số session cho Train, 10% cho Test.
- Gán toàn bộ sự kiện của các Session trước Cutoff vào `train_df`, các Session sau Cutoff vào `test_df`.

### Bước 2: Fit Encoder trên tập Train & Guard Warm-start
- **Item Encoder:** Chỉ `fit` trên `train_df['product_id']`.
- **User Encoder:** Chỉ `fit` trên `train_df['user_id']`.
- **Guard Warm-start Positive (thực hiện SAU khi fit Encoder):** Lọc `test_df` lấy các dòng `label > 0` có `user_id ∈ user_encoder.classes_` VÀ `product_id ∈ item_encoder.classes_`. Nếu số lượng < 100, log Warning và SKIP tính Recall@K. Không tự động thay đổi Cutoff.

### Bước 3: Thuật toán Mix Negative
- **Tập Positive (`pos_df`)**: Từ `train_df` (`label > 0`). GIỮ NGUYÊN trùng lặp (user mua lặp lại = tín hiệu sức mạnh).
- **Tập Hard Negative (`hard_neg_df`)**:
  - Lọc `train_df` (`label = 0`).
  - `drop_duplicates(['user_id', 'product_id'])` để mỗi cặp chỉ giữ 1 dòng.
  - Sample tối đa `len(pos_df) * HARD_NEG_RATIO`. Nếu không đủ thì dùng toàn bộ và log Warning.
- **Tập Easy Negative (Global Random):**
  - Tạo `observed_train_pairs = train_df[['user_id','product_id']].drop_duplicates()`.
  - Vòng lặp `while pool < target_easy_negatives`:
    - Sinh batch cặp `(user_id, product_id)` ngẫu nhiên từ tập User/Item đã có trong Encoder.
    - **Dedup nội bộ pool:** `drop_duplicates(['user_id', 'product_id'])` trên toàn bộ pool đã gom.
    - **Anti-join:** Left-join với `observed_train_pairs`, vứt bỏ các cặp trùng lịch sử.
    - Nối vào pool. Lặp đến khi đủ hoặc chạm `max_attempts`. Log Warning nếu thiếu.

### Bước 4: Khắc phục Focal Loss
- Thêm `RECALL_POS_ALPHA = 0.75` vào `src/config.py`.
- Truyền giá trị qua constructor: `FocalLoss(alpha=config.RECALL_POS_ALPHA, gamma=2.0)`.
- Sửa hàm `forward` trong `trainer.py`:
  ```python
  alpha = torch.as_tensor(self.alpha, device=targets.device, dtype=targets.dtype)
  alpha_t = torch.where(targets == 1, alpha, 1 - alpha)
  focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
  ```
- Không đụng chạm vào `model.py` (giữ nguyên sigmoid).

### Bước 5: Cold-start Serving (Sửa `pipeline.py`)
- Cold-start User không có ID trong Encoder nhưng CÓ `session_items`:
  - Lọc `session_items` theo `product_id_to_idx` (chỉ giữ item hợp lệ trong Encoder).
  - Nếu còn ≥ 1 item hợp lệ: Tính Session Vector → FAISS search → Tạo DataFrame ứng viên với `recalled_by_long_term=0`, `recalled_by_session=1`, các user features = 0 → Chuyển sang Ranker.
  - Nếu không có item hợp lệ hoặc 0 candidate từ FAISS: Popular Fallback.

### Bước 6: Holdout Evaluation & Retrain Ranker
- **Recall@K:** Chỉ tính trên Warm-start (`user_id ∈ user_encoder` VÀ `product_id ∈ item_encoder`, riêng rẽ, không yêu cầu cặp từng xuất hiện cùng nhau).
- **Retrain Ranker (Hạng mục riêng biệt):** `train_on_recall.py` đã được sửa dùng Point-in-Time (`merge_asof`). Sau khi Recall sinh Embedding mới, phải chạy lại Ranker pipeline. Đây là hạng mục bắt buộc, không phải "chỉ rerun".
