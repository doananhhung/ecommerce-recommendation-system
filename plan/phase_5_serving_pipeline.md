# Phase 5: Phục vụ mô hình & Đánh giá (Serving Pipeline)

Mục tiêu: Xây dựng hệ thống suy luận (Inference API) tốc độ cao, kết hợp Vector Search và Ranking thời gian thực.

- [ ] **1. Khởi tạo FAISS Index (Vector Search)**
  - [ ] Trong `faiss_index.py`, load Item Embeddings thu được từ kết quả huấn luyện Phase 3.
  - [ ] Xây dựng index (ví dụ: `IndexFlatIP` cho tích vô hướng hoặc `IndexIVFFlat` cho ANN) để thực hiện quét vector siêu tốc (mili-giây).
- [ ] **2. Cập nhật Lớp Pipeline**
  - [ ] Viết lại hàm `recommend` trong `pipeline.py`.
  - [ ] **Bước 1 (Recall):** Lấy User Embedding tương ứng -> Query vào FAISS để lấy Top-200 Items ứng viên.
  - [ ] **Bước 2 (Feature Fetching):** Trích xuất các Point-in-time features mới nhất của User và 200 Items ứng viên đó.
  - [ ] **Bước 3 (Ranking):** Đưa danh sách Features vào LightGBM (từ Phase 4) để tính điểm (Score).
  - [ ] **Bước 4:** Sort theo Score từ cao xuống thấp và trả về Top-10.
- [ ] **3. Tích hợp API (FastAPI)**
  - [ ] Viết các endpoint trong `main_serve.py` (ví dụ: `POST /recommend`).
  - [ ] Load các mô hình (FAISS, LightGBM) vào bộ nhớ một lần duy nhất lúc khởi động ứng dụng (startup events) để giảm độ trễ.
  - [ ] Test API bằng cURL hoặc Postman để đảm bảo Response Time < 100ms.