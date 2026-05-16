# Phase 4: Mô hình Xếp hạng (Ranking Model - Precision Scoring)

## Stage 4.1: Chuẩn bị Dữ liệu Huấn luyện (Feature Joining)
- [x] Chạy luồng Recall để sinh ra ~200 ứng viên (candidates) cho mỗi User trong tập Train.
- [x] Nối (Join) các đặc trưng của User và Item từ `data/feature_store/` vào danh sách ứng viên này.
- [x] Xử lý Missing values và mã hóa các biến phân loại (Categorical Encoding) sẵn sàng cho Tree-model.

## Stage 4.2: Huấn luyện Mô hình (Model Training)
- [x] Cấu hình LightGBM / XGBoost với chế độ chạy GPU (`device="gpu"`).
- [x] Thực hiện train mô hình để dự đoán xác suất CTR (Click-Through Rate).
- [x] Đánh giá mô hình trên tập validation bằng các hàm đo lường `NDCG@K` và `MRR` (`src/ranking_model/metrics.py`).
- [x] Lưu file mô hình đã train thành công vào `models_store/ranker_model.txt`.
