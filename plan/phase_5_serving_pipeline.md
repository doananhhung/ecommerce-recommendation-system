# Phase 5: Phục vụ Hệ thống & Đánh giá (Serving Pipeline)

## Stage 5.1: Tích hợp Tìm kiếm Vector (FAISS Integration)
- [x] Triển khai `src/serving/faiss_index.py`.
- [x] Khởi tạo FAISS Index (ưu tiên `faiss-gpu`) và nạp `item_embeddings.npy` vào Index.
- [x] Viết hàm truy vấn ANN để lấy Top-200 ứng viên từ 1 vector User bất kỳ.
- [x] Kiểm tra thời gian truy xuất đảm bảo tốc độ cực nhanh (< 50ms).

## Stage 5.2: Kết nối Luồng End-to-End (End-to-End Pipeline)
- [x] Triển khai `src/serving/ranker.py` để nạp mô hình LightGBM từ đĩa lên bộ nhớ.
- [x] Triển khai `src/serving/pipeline.py`. Quy trình: Nhận User ID -> Lấy Vector User -> Truy vấn FAISS (200 items) -> Tra cứu Đặc trưng (Feature Lookup) -> LightGBM Ranker -> Trả về Top 20.
- [x] Viết logic Fallback (Cold-start) cho trường hợp User mới hoàn toàn.

## Stage 5.3: Entrypoints & API
- [x] Hoàn thiện `main_train.py` để tự động hóa toàn bộ luồng Train từ Phase 2 đến Phase 4 chỉ bằng 1 câu lệnh.
- [x] Hoàn thiện `main_serve.py` sử dụng FastAPI để bọc Pipeline thành một RESTful API (VD: `/recommend/{user_id}`).
- [x] Gửi Request giả lập (Load Testing) để kiểm tra độ trễ (Latency) toàn hệ thống và đảm bảo < 100ms.
