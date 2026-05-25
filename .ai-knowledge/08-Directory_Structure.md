# 8. Cấu trúc Thư mục và File (Directory Structure)

Để đảm bảo dự án được tổ chức rõ ràng, dễ bảo trì và dễ dàng mở rộng sang các hệ thống production thực tế, toàn bộ workspace được phân chia theo chuẩn Cấu trúc dự án Machine Learning / MLOps.

## Sơ đồ cấu trúc tổng thể (Tree Structure)

```text
EDA_project/
│
├── .ai-knowledge/          # Chứa các file định nghĩa, triết lý, kiến thức chuẩn của project (Documentation).
├── document/               # Các tài liệu, bản nháp, file Markdown giải thích nghiệp vụ ban đầu.
│
├── data/                   # (Đưa vào .gitignore) Thư mục chứa dữ liệu tĩnh.
│   ├── raw/                # Dữ liệu gốc nguyên bản chưa qua chỉnh sửa (CSV, Parquet...).
│   ├── processed/          # Dữ liệu sau khi đã làm sạch, xử lý missing.
│   ├── sessions/           # Dữ liệu đã được gom nhóm theo phiên (Sessionization).
│   └── feature_store/      # Nơi lưu offline các Vector, Label được tạo ra (Parquet, Numpy array).
│
├── notebooks/              # Giai đoạn nghiên cứu và thử nghiệm (Research & EDA).
│   ├── 01_EDA.ipynb        # Phân tích khám phá dữ liệu, vẽ biểu đồ.
│   ├── 02_Feature_Eng.ipynb# Thử nghiệm các logic tạo đặc trưng, gán nhãn giả.
│   └── 03_Model_Exp.ipynb  # Train thử nhỏ lẻ các mô hình để check code trước khi đưa vào src/.
│
├── src/                    # Mã nguồn chính của toàn bộ chương trình (Production Code).
│   ├── __init__.py
│   ├── config.py           # Chứa các hằng số, đường dẫn, siêu tham số (Hyperparameters).
│   │
│   ├── data_pipeline/      # Lớp xử lý dữ liệu (Pre-processing & ETL).
│   │   ├── sessionizer.py  # Code gom nhóm hành vi theo thời gian thành Session.
│   │   ├── pseudo_label.py # Logic gán nhãn giả (0 từ lack of click, 1 từ click).
│   │   └── featurizer.py   # Chuyển đổi dữ liệu thành các trường đặc trưng/Vector hóa.
│   │
│   ├── recall_model/       # Giai đoạn 1: Truy xuất ứng viên (Candidate Generation).
│   │   ├── dataset.py      # PyTorch Dataset/DataLoader để train mô hình tạo nhúng (Embedding).
│   │   ├── model.py        # Kiến trúc mô hình PyTorch (Matrix Factorization / Two-Tower).
│   │   └── trainer.py      # Logic huấn luyện (Loss, Optimizer, Epochs) cho Recommender.
│   │
│   ├── ranking_model/      # Giai đoạn 2: Xếp hạng tinh chỉnh.
│   │   ├── lgbm_train.py   # Script load tập đặc trưng kết hợp (User-200 Items) để train LightGBM.
│   │   └── metrics.py      # Định nghĩa các hàm tính NDCG, MRR, HR@K để đánh giá lúc train xếp hạng.
│   │
│   └── serving/            # Lớp Phục vụ (Inference Pipeline) kết nối từ end-to-end.
│       ├── faiss_index.py  # Logic tạo, lưu, load, và truy vấn gần đúng (ANN) trên tệp vector Item bằng FAISS.
│       ├── ranker.py       # Load model LightGBM để predict score từ list 200 items.
│       └── pipeline.py     # Gộp toàn bộ luồng User ID -> Recall FAISS -> Feature Lookup -> Ranker -> Top 20.
│
├── models_store/           # (Đưa vào .gitignore) Thư mục lưu các artifacts (Cân nặng) của mô hình.
│   ├── recall_weights.pth  # Trọng số PyTorch mô hình Two-Tower/MF.
│   └── ranker_model.txt    # File model serialize của LightGBM.
│
├── main_train.py           # Entrypoint kích hoạt luồng tự động Train lại toàn bộ hệ thống từ dữ liệu mới.
├── main_serve.py           # Entrypoint chạy API (FastAPI) phục vụ gợi ý thời gian thực.
├── pyproject.toml          # Danh sách dependency và cấu hình gói phụ thuộc của dự án (sử dụng uv).
├── uv.lock                 # File khóa môi trường đồng nhất và bảo mật (sinh bởi uv).
└── README.md               # Giới thiệu sơ bộ dự án, cách setup, cách chạy lệnh train/serve.
```

## Giải thích chức năng thư mục quan trọng

1. **`notebooks/` vs `src/`:** `notebooks/` chỉ dùng để mò mẫm vẽ biểu đồ tìm hiểu dữ liệu ban đầu. Khi code chạy đúng, ngay lập tức chuyển logic thành các hàm chuẩn (functions/classes) đặt tại `src/` để tái sử dụng xuyên suốt hệ thống.
2. **`models_store/`:** Rất quan trọng khi chạy Offline Pipeline. Mô hình Train xong ở `.py` scripts sẽ lưu các trạng thái, bộ nén vector vào đây, sau đó luồng API ở `main_serve` sẽ load file từ thư mục này lên RAM để phục vụ.
3. **`data/feature_store/`:** Đóng gói các bảng tính toán xong dưới dạng file Parquet và tệp `item_embeddings.npy` của mô hình Recall. Khi chạy server API, chỉ mục FAISS sẽ được dựng trực tiếp trên RAM từ tệp `.npy` này giúp tối ưu hóa I/O và tránh lỗi xung đột hệ điều hành đối với tệp `.index` nhị phân.