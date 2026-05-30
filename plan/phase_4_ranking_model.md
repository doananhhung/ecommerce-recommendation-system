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

- [x] **6. Triển khai Giải pháp 1: Huấn luyện trên Danh sách Ứng viên Recall thực tế (Train-on-Recall)**
  - [x] **6.1. Xây dựng Script Sinh dữ liệu huấn luyện Ranking từ Recall Candidates:**
    - Tạo script `src/ranking_model/train_on_recall.py`.
    - Tải sẵn `RecommendationPipeline` từ khâu Serving để tái sử dụng toàn bộ logic triệu hồi 2 kênh song song kết hợp FAISS.
    - Với mỗi nhóm phiên tương tác thực tế của người dùng `(user_id, custom_session_id)` trong `labeled_sessions.parquet`:
      - Truy vấn bộ Recall lấy 200 ứng viên (bao gồm các chỉ thị kênh `recalled_by_long_term`, `recalled_by_session` thực tế từ FAISS).
      - Xác định các sản phẩm thực tế người dùng tương tác trong phiên này làm tập tương tác thực tế (`true_interactions`).
      - Tạo các bản ghi mẫu âm (Negative Samples) từ các ứng viên được recall nhưng người dùng KHÔNG tương tác trong phiên. Gán nhãn `0` và `sample_weight = 0.1`.
      - Các ứng viên trùng với tương tác thực tế sẽ được gán nhãn thực tế (`view` = 0, `cart/purchase` = 1) và `sample_weight` tương ứng.
      - Ghép nối các đặc trưng User/Item tĩnh và động.
  - [x] **6.2. Chia dữ liệu Train/Test và Huấn luyện LightGBM:**
    - Áp dụng Time-based Splitting 80/20 nghiêm ngặt để chia tập dữ liệu Ranking mới.
    - Huấn luyện LightGBM Binary Classifier với `sample_weight` và cấu hình mất cân bằng.
  - [x] **6.3. Đánh giá chất lượng xếp hạng sau cải tiến:**
    - Tính toán Validation AUC, NDCG@10 và MRR trên tập kiểm thử thực tế.
    - Kiểm nghiệm sự cải thiện vượt bậc của các chỉ số xếp hạng so với phiên bản baseline.
  - [x] **6.4. Đồng bộ hóa và cập nhật mô hình:**
    - Lưu mô hình xếp hạng mới cải tiến vào `models_store/ranker_model.txt`.