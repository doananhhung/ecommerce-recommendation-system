# Two-Stage Recommender System

Hệ thống gợi ý sản phẩm Two-Stage (Recall & Ranking) sử dụng PyTorch và LightGBM cho bài toán thương mại điện tử.

## Cấu trúc dự án
Xem chi tiết tại `.ai-knowledge/11-Project_Architecture.md`.

## Yêu cầu môi trường
- Python 3.10+
- Quản lý package bằng `uv` (Hoặc dùng `pip install -r requirements.txt` nếu tự xuất requirements)
- CUDA-enabled GPU (Khuyến nghị cho huấn luyện PyTorch và LightGBM)

## Cách chạy dự án (End-to-End)

1. Cài đặt các thư viện:
   ```bash
   uv sync
   ```

2. Tải tập dữ liệu hành vi (e.g. `2019-Oct.csv` từ REES46) và đặt vào `data/raw/2019-Oct.csv`.

3. Chạy luồng huấn luyện toàn bộ hệ thống (Data Pipeline -> Recall -> Ranking):
   ```bash
   uv run python main_train.py
   ```
   *Lưu ý: Quá trình này sẽ tự động lưu feature store, vector embeddings và models vào `data/` và `models_store/`.*

4. Khởi động API Server để phục vụ dự đoán thời gian thực:
   ```bash
   uv run python main_serve.py
   ```
   Server sẽ chạy tại `http://127.0.0.1:8000`.

## API Endpoints
- **GET `/recommend/{user_id}`**: Lấy top 20 sản phẩm gợi ý cho User.
  - Ví dụ test bằng cURL:
    ```bash
    curl -X GET "http://127.0.0.1:8000/recommend/512345678"
    ```
