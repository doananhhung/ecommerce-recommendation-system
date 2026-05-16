# Phase 1: Phân tích Dữ liệu (EDA) & Thiết lập

Mục tiêu: Thiết lập môi trường và nhận diện các đặc trưng, vấn đề cốt lõi của tập dữ liệu hành vi thương mại điện tử.

- [x] **1. Thiết lập môi trường dự án**
  - [x] Tạo `uv.lock` hoặc `requirements.txt` (nếu có).
  - [x] Cài đặt thư viện: `pandas`, `numpy`, `lightgbm`, `torch`, `faiss`, `fastapi`.
- [x] **2. Tải và đọc dữ liệu thô**
  - [x] Đọc một phần dữ liệu `2019-Oct.csv` để tránh tràn RAM.
- [x] **3. Phân tích Dữ liệu Khám phá (EDA)**
  - [x] Kiểm tra phân phối `event_type` (view, cart, purchase). Phát hiện mất cân bằng dữ liệu cực đoan (~96.8% view).
  - [x] Tính toán độ thưa thớt (Sparsity) của ma trận User-Item (~99.99%).
  - [x] Nhận diện giá trị khuyết thiếu ở `category_code` và `brand`.
  - [x] Vẽ đồ thị phân phối giá (`price`) và phát hiện phân phối lệch phải (skewed).