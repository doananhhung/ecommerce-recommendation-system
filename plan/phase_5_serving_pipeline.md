# Phase 5: Phục vụ mô hình & Đánh giá (Serving Pipeline)

Mục tiêu: Xây dựng hệ thống suy luận (Inference API) tốc độ cao, kết hợp Vector Search và Ranking thời gian thực.

- [x] **1. Khởi tạo FAISS Index (Vector Search)**
  - [x] Trong `faiss_index.py`, load Item Embeddings thu được từ kết quả huấn luyện Phase 3.
  - [x] Xây dựng index (ví dụ: `IndexFlatIP` cho tích vô hướng hoặc `IndexIVFFlat` cho ANN) để thực hiện quét vector siêu tốc (mili-giây).
- [x] **2. Cập nhật Lớp Pipeline**
  - [x] Viết lại hàm `recommend` trong `pipeline.py`.
  - [x] **Bước 1 (Recall):** Lấy User Embedding tương ứng -> Query vào FAISS để lấy Top-200 Items ứng viên.
  - [x] **Bước 2 (Feature Fetching):** Trích xuất các Point-in-time features mới nhất của User và 200 Items ứng viên đó.
  - [x] **Bước 3 (Ranking):** Đưa danh sách Features vào LightGBM (từ Phase 4) để tính điểm (Score).
  - [x] **Bước 4:** Sort theo Score từ cao xuống thấp và trả về Top-10.
- [x] **3. Tích hợp API (FastAPI)**
  - [x] Viết các endpoint trong `main_serve.py` (ví dụ: `POST /recommend`).
  - [x] Load các mô hình (FAISS, LightGBM) vào bộ nhớ một lần duy nhất lúc khởi động ứng dụng (startup events) để giảm độ trễ.
  - [x] Test API bằng cURL hoặc Postman để đảm bảo Response Time < 100ms.
- [x] **4. Đồng bộ hóa Đặc trưng (Feature Alignment & Single Source of Truth)**
  - [x] Tập trung cấu hình đặc trưng số học và phân loại trong `config.py` làm nguồn chuẩn duy nhất.
  - [x] Cập nhật hàm `extract_item_features` để lưu trữ dữ liệu phân loại (`category_code`, `brand`) của sản phẩm vào Serving Feature Store.
  - [x] Nâng cấp `LightGBMRanker.predict()` tự động đọc danh sách đặc trưng từ tệp mô hình đã load và bù đắp các cột bị thiếu, ngăn chặn lỗi crash không đồng nhất đặc trưng giữa train/serving.