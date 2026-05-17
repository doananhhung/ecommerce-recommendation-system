# 3. Stack Công nghệ (Tech Stack)

Đây là Stack được áp dụng cho môi trường Batch/Static Dataset hiện tại.

## 1. Data Processing: Pandas / Apache Spark
- **Mục đích:** Đọc tập dữ liệu (CSV/Parquet), tiền xử lý, trích xuất đặc trưng (Feature Engineering), One-hot encoding, tạo nhãn giả (Pseudo-labeling).
- **So sánh:**
  - *Tại sao không dùng Flink (như trong báo cáo ĐH)?* Flink sinh ra để xử lý Stream (Push), tốn nhiều tài nguyên setup. Với tập dữ liệu Offline (đã đứng yên trên đĩa), Pandas/Spark xử lý trọn gói toàn bộ mà không lo vấn đề Lag/Late Data.

## 2. Vector Search (ANN): FAISS (hoặc ScaNN)
- **Mục đích:** Lưu trữ hàng triệu vector sản phẩm để truy vấn không gian Nearest Neighbor trong giai đoạn Recall.
- **Tại sao sử dụng:** Thống kê độ tương đồng toán học (Cosine similarity) bằng thuật toán quét toàn bộ (Brute-force) $O(N)$ là không khả thi. FAISS giúp thực thi truy vấn $O(\log N)$.
- **Hậu quả nếu bỏ qua:** Thời gian dự đoán (Inference time) nhảy vọt lên vài giây đến vài phút mỗi người dùng thay vì < 100ms.

## 3. Deep Learning Framework: PyTorch 2.7+
- **Mục đích:** Xây dựng mô hình nhúng (Embedding) như Matrix Factorization, Two-Tower Model để học biểu diễn người dùng và sản phẩm trong không gian Latent Space.

## 4. Tree-based Model: LightGBM / XGBoost
- **Mục đích:** Mô hình Ranking xếp thứ tự đầu ra dựa trên vô số các feature giao cắt.
- **So sánh:**
  - *Tại sao không dùng Deep Learning cho Ranking?* Dữ liệu dạng bảng (Tabular data) có nhiều trường Categorical thưa thớt. LightGBM được chứng minh thực nghiệm là mạnh hơn, cấu trúc node/leaf xử lý Missing Value tốt và tính toán thực thi siêu nhẹ.

## 5. Feature Store (Khái niệm)
- **Mục đích:** Là nơi lưu trữ Offline store (S3/Disk) và Online Store (Redis) cho vector. Đảm bảo tính nhất quán (Feature Consistency) không bị skew giữa lúc Train và lúc Predict.