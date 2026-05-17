# 12. Kiến trúc và Cấu trúc tệp tin dự án (Project Architecture Documentation)

Tài liệu này giải thích chi tiết mục đích và cách thực hiện (implementation) của từng thư mục và tệp tin trong hệ thống Two-Stage Recommender System.

## 1. Thư mục gốc (Root Directory)

- `main_train.py`:
  - **Mục đích:** Entry point để chạy toàn bộ luồng huấn luyện offline tự động (từ dữ liệu thô ra mô hình cuối cùng).
  - **Cách thực hiện:** Sử dụng thư viện `subprocess.run` để gọi lần lượt các script Python của từng giai đoạn: `data_pipeline`, `recall_model`, và `ranking_model` theo thứ tự tuyến tính.

- `main_serve.py`:
  - **Mục đích:** Entry point để khởi chạy API server phục vụ truy vấn và gợi ý sản phẩm theo thời gian thực (Real-time serving).
  - **Cách thực hiện:** Khởi tạo ứng dụng `FastAPI`. Trong event `startup`, nó nạp sẵn các mô hình, FAISS index, và kho đặc trưng vào bộ nhớ RAM thông qua lớp `RecommendationPipeline`. Phơi bày API endpoint `/recommend/{user_id}` cho phía client gọi.

- `pyproject.toml` / `uv.lock`:
  - **Mục đích:** Quản lý môi trường, siêu dữ liệu dự án và các gói phụ thuộc (dependencies) bằng trình quản lý `uv`.

## 2. Thư mục `src/` (Mã nguồn Cốt lõi)

Đây là nơi chứa toàn bộ logic Production của hệ thống, chia thành 4 phân hệ chính.

### 2.1. `src/data_pipeline/` (Xử lý dữ liệu)
- `run_pipeline.py`:
  - **Mục đích:** Orchestrator (người điều phối) điều khiển chu trình ETL dữ liệu.
  - **Cách thực hiện:** Đọc file CSV, gọi tuần tự `create_sessions()`, `extract_user_features()`, `extract_item_features()`, và `apply_pseudo_labels()`. Ghi kết quả cuối cùng ra định dạng `.parquet` tối ưu hóa I/O.
- `sessionizer.py`:
  - **Mục đích:** Gom nhóm các hành vi rời rạc của User thành các phiên (Sessions).
  - **Cách thực hiện:** Sắp xếp dữ liệu theo thời gian, tính sự chênh lệch thời gian giữa các hành vi liên tiếp. Nếu khoảng cách vượt qua ngưỡng, chia tách thành `custom_session_id` mới.
- `featurizer.py`:
  - **Mục đích:** Trích xuất đặc trưng cho user và item (Feature Engineering).
  - **Cách thực hiện:** Sử dụng các hàm `groupby` và aggregation để đếm tổng lượt tương tác, đếm số user độc nhất. *(Lưu ý: Mã nguồn hiện tại đang cần tái cấu trúc lại tính năng Point-in-time calculation để chống lỗi rò rỉ dữ liệu tương lai).*
- `pseudo_label.py`:
  - **Mục đích:** Gán nhãn giả (Pseudo-label) dựa trên hành vi ẩn (Implicit Feedback) để huấn luyện.
  - **Cách thực hiện:** Chuyển đổi các sự kiện mua sắm: gán các event `view` thành nhãn 0, `cart`/`purchase` thành nhãn 1.

### 2.2. `src/recall_model/` (Giai đoạn Truy xuất Ứng viên)
- `run_recall.py`:
  - **Mục đích:** Tập lệnh huấn luyện Giai đoạn 1.
  - **Cách thực hiện:** Label Encoding các User_ID/Product_ID chuỗi thành số nguyên Index, khởi tạo DataLoader, thiết lập kiến trúc mạng PyTorch, chạy Epochs và lưu các trọng số `recall_weights.pth` kèm theo `LabelEncoders` ra đĩa.
- `dataset.py`:
  - **Mục đích:** PyTorch `Dataset` custom xử lý dữ liệu tương tác.
  - **Cách thực hiện:** Nhận pandas series index, trả về các cặp tensor (user_idx, item_idx, label) phục vụ DataLoader.
- `model.py`:
  - **Mục đích:** Định nghĩa kiến trúc Mạng Nơ-ron phân rã ma trận (Matrix Factorization).
  - **Cách thực hiện:** Sử dụng `nn.Embedding` cho người dùng và sản phẩm. Dự đoán bằng tính tích vô hướng (Dot product) của 2 vector và kích hoạt bằng `Sigmoid`.
- `trainer.py`:
  - **Mục đích:** Trừu tượng hóa vòng lặp huấn luyện PyTorch.
  - **Cách thực hiện:** Định nghĩa hàm Loss (`BCELoss`), bộ tối ưu (Adam). Chạy `forward pass`, tính backpropagation và update vector embedding qua từng epoch.

### 2.3. `src/ranking_model/` (Giai đoạn Xếp hạng)
- `run_ranking.py`:
  - **Mục đích:** Tập lệnh huấn luyện mô hình Cây quyết định Giai đoạn 2.
  - **Cách thực hiện:** Nối (Merge) tập data nhãn với file Parquet feature store. Chia train/test (hiện đang dùng `train_test_split`), khởi tạo `RankingTrainer` và lưu ra `ranker_model.txt`.
- `lgbm_train.py`:
  - **Mục đích:** Lớp Wrapper đóng gói quy trình của thuật toán `LightGBM`.
  - **Cách thực hiện:** Đóng gói Pandas DataFrame thành đối tượng `lgb.Dataset`. Truyền các tham số cấu hình Hyper-parameters và gọi API huấn luyện.
- `metrics.py`:
  - **Mục đích:** Tính toán các độ đo chất lượng xếp hạng riêng biệt cho Recommender System.
  - **Cách thực hiện:** Triển khai công thức tính `NDCG@K` và `MRR@K` từ mảng xác suất dự đoán so với nhãn thực tế.

### 2.4. `src/serving/` (Đường ống Phục vụ API)
- `pipeline.py`:
  - **Mục đích:** Trung tâm kết nối (Glue code) liên kết End-to-End quá trình nội suy (Inference).
  - **Cách thực hiện:** Class `RecommendationPipeline` chứa logic nạp mô hình vào RAM. Quy trình nhận: (1) `user_id`, (2) Load user embedding từ PyTorch bằng index, (3) Lấy top K ứng viên từ thư viện FaissSearcher, (4) Nối feature từ DataFrame cache, (5) Chấm điểm bằng Ranker, và (6) Trả về JSON list.
- `faiss_index.py`:
  - **Mục đích:** Quản lý không gian Vector Index.
  - **Cách thực hiện:** Đọc tệp numpy `item_embeddings.npy`, bao bọc thành lớp `IndexFlatIP` của thư viện `faiss` chuyên xử lý tìm kiếm lân cận gần nhất (Approximate Nearest Neighbors).
- `ranker.py`:
  - **Mục đích:** Quản lý tiến trình suy luận (predict) của mô hình Giai đoạn 2.
  - **Cách thực hiện:** Nạp mô hình `.txt` của LightGBM, nhận đầu vào là dataframe pandas gồm các features và trả về list Float dự đoán phân loại.

### 2.5. Các thư mục hỗ trợ
- `.ai-knowledge/`: Chứa các Markdown chuẩn mực, định hướng dự án, nhật ký lỗi (Bug tracker) giúp nhà phát triển nắm bắt được kiến thức chuyên môn của mô hình mà không cần đọc code.
- `plan/`: Chứa các check-list, kế hoạch phát triển từng Phase. Các markdown file này ghi lại thiết kế toán học, phương hướng fix lỗi hiện tại.
- `data/`: Thư mục lưu dữ liệu thô (raw), dữ liệu phân tích, parquet và vector được bỏ qua trong Git.
- `models_store/`: Lưu artifacts như mô hình cây, weights NN, và thư viện Encoder trung gian.
- `notebooks/`: Chứa Jupyter Notebook để Data Scientist khám phá và chạy Exploratory Data Analysis (EDA) ban đầu.