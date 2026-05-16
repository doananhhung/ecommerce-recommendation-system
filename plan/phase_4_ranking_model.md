# Phase 4: Giai đoạn Xếp hạng (Ranking Model)

Mục tiêu: Sử dụng cây quyết định Gradient Boosting (LightGBM) để chấm điểm và xếp hạng lại Top-K ứng viên từ mô hình Recall.

- [ ] **1. Cấu hình LightGBM cho mất cân bằng dữ liệu**
  - [ ] Trong `lgbm_train.py`, bổ sung tham số `is_unbalance=True` hoặc tự tính toán cấu hình `scale_pos_weight` vào `params` huấn luyện.
- [ ] **2. Tích hợp Trọng số Mẫu (Sample Weights)**
  - [ ] Truyền trọng số (view=0.1, cart=0.5, purchase=1.0) từ Data Pipeline vào cấu trúc `lgb.Dataset(..., weight=sample_weights)` để mô hình hiểu mức độ quan trọng của từng tương tác.
- [ ] **3. Huấn luyện Mô hình Xếp hạng**
  - [ ] Tải các đặc trưng Point-in-time đã sinh ra từ Phase 2.
  - [ ] Đảm bảo sử dụng Time-based Splitting (trong `run_ranking.py`) để tạo `X_train`, `X_test`. Tuyệt đối không xáo trộn dữ liệu tương lai.
  - [ ] Huấn luyện LightGBM (`num_boost_round=100` hoặc Early Stopping).
- [ ] **4. Đánh giá Mô hình**
  - [ ] Cập nhật và sử dụng độ đo `NDCG@K` và `MRR@K` trong file `metrics.py`.
  - [ ] Đo lường hiệu suất mô hình trên tập Test.
  - [ ] Lưu mô hình xuống đĩa cứng (`ranker_model.txt`).