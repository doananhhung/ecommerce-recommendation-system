# Phase 4: Giai đoạn Xếp hạng (Ranking Model)

Mục tiêu: Sử dụng cây quyết định Gradient Boosting (LightGBM) để chấm điểm và xếp hạng lại Top-K ứng viên từ mô hình Recall.

- [x] **1. Cấu hình LightGBM cho mất cân bằng dữ liệu**
  - [x] Trong `lgbm_train.py`, bổ sung tham số `is_unbalance=True` hoặc tự tính toán cấu hình `scale_pos_weight` vào `params` huấn luyện.
- [x] **2. Tích hợp Trọng số Mẫu (Sample Weights)**
  - [x] Truyền trọng số (view=0.1, cart=0.5, purchase=1.0) từ Data Pipeline vào cấu trúc `lgb.Dataset(..., weight=sample_weights)` để mô hình hiểu mức độ quan trọng của từng tương tác.
- [x] **3. Huấn luyện Mô hình Xếp hạng**
  - [x] Tải các đặc trưng Point-in-time đã sinh ra từ Phase 2.
  - [x] Đảm bảo sử dụng Time-based Splitting (trong `run_ranking.py`) để tạo `X_train`, `X_test`. Tuyệt đối không xáo trộn dữ liệu tương lai.
  - [x] Huấn luyện LightGBM (`num_boost_round=100` hoặc Early Stopping).
- [x] **4. Đánh giá Mô hình**
  - [x] Cập nhật và sử dụng độ đo `NDCG@K` và `MRR@K` trong file `metrics.py`.
  - [x] Đo lường hiệu suất mô hình trên tập Test.
  - [x] Lưu mô hình xuống đĩa cứng (`ranker_model.txt`).
- [x] **5. Đồng bộ hóa và Huấn luyện Đặc trưng Động**
  - [x] Thống nhất danh sách đặc trưng số và phân loại tại `config.py` để tránh phân tán.
  - [x] Cập nhật `run_ranking.py` để nạp động đặc trưng từ `config.py` thay vì hardcode cứng danh sách cũ, giúp mô hình tự học các đặc trưng phiên mới (`user_session_interaction_count`, `item_session_popularity`, `recalled_by_long_term`, `recalled_by_session`).