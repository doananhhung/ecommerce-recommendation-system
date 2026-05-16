# Phase 3: Giai đoạn Triệu hồi (Recall Model)

Mục tiêu: Sàng lọc nhanh từ hàng triệu sản phẩm xuống hàng trăm ứng viên tiềm năng bằng cách sử dụng Deep Learning/Matrix Factorization.

- [ ] **1. Xây dựng Data Loader với Negative Sampling**
  - [ ] Trong `dataset.py`, triển khai cơ chế Negative Sampling ngẫu nhiên (ví dụ: lấy 4 view cho mỗi 1 purchase/cart) để cân bằng tỷ lệ mẫu.
- [ ] **2. Cải tiến Cấu trúc Mạng Neural (PyTorch)**
  - [ ] Khởi tạo mô hình Matrix Factorization (hoặc Two-Tower) với lớp User/Item Embeddings.
  - [ ] Tích hợp **Focal Loss** vào `trainer.py` thay cho `BCELoss` cơ bản để ép mô hình tập trung vào các mẫu thiểu số (cart, purchase).
  - [ ] (Tuỳ chọn nâng cao) Tích hợp hàm **BPR Loss (Bayesian Personalized Ranking)** để học xếp hạng theo cặp, giúp giải quyết bài toán ma trận thưa thớt 99.99%.
- [ ] **3. Huấn luyện và Đánh giá Recall**
  - [ ] Huấn luyện mô hình trên tập Train (được chia theo thời gian ở Phase 2).
  - [ ] Đánh giá độ phủ (Hit Rate / Recall@K) trên tập Validation/Test.
  - [ ] Trích xuất và lưu trọng số Embeddings của User và Item để phục vụ cho Ranking và Serving (`recall_weights.pth`).