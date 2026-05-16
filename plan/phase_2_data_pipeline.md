# Phase 2: Luồng dữ liệu & Trích xuất Đặc trưng (Data Pipeline)

## Stage 2.1: Thử nghiệm Logic (Prototyping - `02_Feature_Eng.ipynb`)
- [x] Viết logic Gom nhóm Phiên (Sessionization) với cửa sổ thời gian 30 phút.
- [x] Viết logic Gán nhãn giả (Pseudo-labeling: 0 cho view, 1 cho cart/purchase).
- [x] Trích xuất các đặc trưng của User (User Features: Tổng lượt click, Category yêu thích...).
- [x] Trích xuất các đặc trưng của Item (Item Features: Lượt xem, Mức giá trung bình...).

## Stage 2.2: Đưa vào Mã nguồn Chính (Productionizing Code)
- [x] Triển khai `src/data_pipeline/sessionizer.py`.
- [x] Triển khai `src/data_pipeline/pseudo_label.py`.
- [x] Triển khai `src/data_pipeline/featurizer.py`.
- [x] Viết script hoặc notebook để xử lý toàn bộ tập dữ liệu Raw (hàng triệu dòng).
- [x] Lưu kết quả các bảng đặc trưng (Feature Store) xuống thư mục `data/feature_store/` định dạng Parquet.
