# Phase 3: Giai đoạn Triệu hồi (Recall Model)

Mục tiêu: Sàng lọc nhanh từ hàng triệu sản phẩm xuống hàng trăm ứng viên tiềm năng bằng cách sử dụng Deep Learning/Matrix Factorization.

- [x] **1. Xây dựng Data Loader với Negative Sampling**
  - [x] Triển khai cơ chế Negative Sampling ngẫu nhiên (lấy view theo tỷ lệ so với purchase/cart) để cân bằng mẫu. (Lưu ý: Được thực hiện tại `run_recall.py` thay vì `dataset.py`).
- [x] **2. Cải tiến Cấu trúc Mạng Neural (PyTorch)**
  - [x] Khởi tạo mô hình Matrix Factorization (hoặc Two-Tower) với lớp User/Item Embeddings.
  - [x] Tích hợp **Focal Loss** vào `trainer.py` thay cho `BCELoss` cơ bản để ép mô hình tập trung vào các mẫu thiểu số (cart, purchase).
  - [ ] (Tuỳ chọn nâng cao) Tích hợp hàm **BPR Loss (Bayesian Personalized Ranking)** để học xếp hạng theo cặp, giúp giải quyết bài toán ma trận thưa thớt 99.99%.
- [x] **3. Huấn luyện và Đánh giá Recall**
  - [x] Huấn luyện mô hình trên tập Train (được chia theo thời gian ở Phase 2).
  - [x] Đánh giá độ phủ (Hit Rate / Recall@K) trên tập Validation/Test (Đã viết code tính Recall@50 tại `run_recall.py`).
  - [x] Trích xuất và lưu trọng số Embeddings của User và Item để phục vụ cho Ranking và Serving (`recall_weights.pth`).
- [x] **4. Triển khai Hệ thống Triệu hồi Hai Kênh (Dual-Channel Recall)**
  - [x] Triển khai Kênh 1 (Long-term preference) dựa trên User Embedding tìm kiếm lân cận gần nhất qua FAISS.
  - [x] Triển khai Kênh 2 (Current session needs) dựa trên việc tính Vector trung bình của các sản phẩm tương tác gần nhất trong phiên để truy vấn FAISS.
  - [x] Gộp ứng viên từ cả 2 kênh và tạo cột chỉ thị `recalled_by_long_term`, `recalled_by_session` bổ trợ thông tin xếp hạng cho LightGBM Ranker.