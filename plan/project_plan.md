# Kế hoạch Tổng thể: Hệ thống Gợi ý Hai Giai đoạn (Two-Stage Recommender System)

Bản kế hoạch này đã được nâng cấp dựa trên các phân tích chuyên sâu về dữ liệu (EDA), khắc phục các lỗi nghiêm trọng về cấu trúc (Data Leakage) và tối ưu hóa toán học cho thuật toán Học máy.

Dự án được chia thành 5 Giai đoạn (Phases) cốt lõi.

---

## 1. Phase 1: Phân tích Dữ liệu (EDA) & Thiết lập (Đã hoàn thành)
- Phân tích cấu trúc file dữ liệu hành vi thương mại điện tử (eCommerce behavior data).
- Nhận diện sự mất cân bằng dữ liệu cực đoan giữa `view` (96.8%) và `purchase/cart` (3.2%).
- Xác định sự thưa thớt của ma trận User-Item (~99.99%).
- Nhận diện các giá trị khuyết thiếu ở `category_code` và `brand`.
- Nhận diện phân phối ngoại lai (skewed) của trường `price`.

---

## 2. Phase 2: Data Pipeline & Point-in-time Feature Engineering
*Mục tiêu: Chuyển đổi dữ liệu thô thành các đặc trưng chất lượng cao, tuyệt đối ngăn chặn rò rỉ dữ liệu tương lai (Data Leakage).*

**2.1. Xử lý Rò rỉ dữ liệu (Chống Time Travel):**
- **Trình tự Thời gian:** Sắp xếp toàn bộ dữ liệu theo `event_time`.
- **Time-based Splitting:** Xây dựng cơ chế chia Train/Validation/Test chặt chẽ dựa trên mốc thời gian (ví dụ: Train trên tuần 1-3, Test trên tuần 4). Nghiêm cấm sử dụng `train_test_split` ngẫu nhiên.
- **Point-in-time Features:** Viết lại `featurizer.py`. Tính toán `user_total_interactions`, `item_total_interactions` bằng phương pháp tích lũy (Cumulative Sum) hoặc Cửa sổ trượt (Rolling Window) chỉ tính đến thời điểm $t$ trước khi xảy ra sự kiện. Không sử dụng `groupby` gộp toàn bộ tập dữ liệu.

**2.2. Khắc phục Dữ liệu khuyết thiếu (Missing Value) & Ngoại lai:**
- **Categorical Imputation:** Không dùng hàm `fillna(0)`. Tạo Vector định danh `<UNKNOWN>` riêng biệt cho các trường `brand` và `category_code` bị thiếu để bảo toàn cấu trúc phân loại. Áp dụng Label Encoding / Target Encoding.
- **Numerical Scaling:** Áp dụng phép biến đổi logarit (`Log Transform: log(x+1)`) cho trường `price` để giảm độ lệch phải, tránh bùng nổ Gradient (Exploding Gradients).

**2.3. Cải tiến Trọng số Phản hồi Ngầm (Implicit Feedback):**
- Gán trọng số tùy chỉnh (Custom Sample Weights) cho từng hành vi thay vì nhị phân hóa: `view = 0.1`, `cart = 0.5`, `purchase = 1.0`.

---

## 3. Phase 3: Giai đoạn Triệu hồi (Recall Model)
*Mục tiêu: Sàng lọc nhanh từ hàng triệu sản phẩm xuống hàng trăm ứng viên tiềm năng.*

**3.1. Xây dựng Data Loader:**
- Triển khai **Negative Sampling** linh hoạt: Lấy mẫu âm tính có chủ đích để cân bằng tỷ lệ mẫu (ví dụ: 1 positive : 4 negatives) giúp giảm nhiễu.

**3.2. Cải tiến Cấu trúc Mạng Neural (PyTorch):**
- Thay vì sử dụng hàm `BCELoss` thông thường dễ bị lớp đa số áp đảo, thay thế bằng các thuật toán chuyên dụng:
  - **Focal Loss:** Điều chỉnh biên độ Gradient động, ép mạng Nơ-ron tập trung tối ưu các mẫu khó (purchase/cart) và giảm sự quan tâm đối với các mẫu dễ (`view`).
  - **Bayesian Personalized Ranking (BPR Loss):** Chuyển bài toán sang học xếp hạng theo cặp (Pairwise Learning). Tối ưu hóa nguyên tắc: "Điểm của sản phẩm được click phải cao hơn điểm của sản phẩm chưa click". Giúp đối phó với độ thưa thớt 99.99% của ma trận User-Item.

---

## 4. Phase 4: Giai đoạn Xếp hạng (Ranking Model)
*Mục tiêu: Sử dụng Cây quyết định (LightGBM) để chấm điểm chính xác top 200 ứng viên từ Recall.*

**4.1. Khắc phục Mất cân bằng dữ liệu cho LightGBM:**
- Bật cấu hình tự động bù đắp trọng số bằng tham số `is_unbalance=True` hoặc tự cấu hình `scale_pos_weight` dựa trên phân phối của tập Train.
- Tích hợp `sample_weight` lấy từ Data Pipeline (đã gán 0.1, 0.5, 1.0) vào Dataset của LightGBM để mô hình hiểu được "mức độ quan tâm".

**4.2. Huấn luyện & Đánh giá (Evaluation):**
- Sử dụng các đặc trưng Point-in-time đã tạo. Đảm bảo Train và Validation sets được tách biệt đúng thời gian.
- Đánh giá bằng các độ đo Ranking chuẩn: NDCG@K, MRR@K.

---

## 5. Phase 5: Hệ thống Phục vụ (Serving Pipeline & Deployment)
*Mục tiêu: Xây dựng API tốc độ cao cho Inference.*

**5.1. Vector Search:**
- Trích xuất User Embeddings và Item Embeddings từ mô hình Recall (Matrix Factorization) sau khi huấn luyện bằng BPR/Focal Loss.
- Đưa Item Embeddings vào FAISS Index để tìm kiếm lân cận gần nhất (Approximate Nearest Neighbors) trong vòng vài mili-giây.

**5.2. API Integration:**
- FastAPI nhận Request (user_id).
- Hệ thống gọi FAISS lấy Top-K (Recall) -> Truy xuất Features từ Feature Store (được cập nhật real-time) -> Đưa vào LightGBM (Ranking) -> Trả về danh sách cuối cùng.

**5.3. Đồng bộ hóa Đặc trưng & Tối ưu hóa Phục vụ (Feature Alignment & Serving Robustness):**
- Tập trung cấu hình danh sách đặc trưng số và phân loại tại `config.py` để làm nguồn chuẩn duy nhất.
- Cập nhật pipeline để lưu trữ đầy đủ các cột categorical (`category_code`, `brand`) vào Serving Feature Store.
- Cải tiến LightGBM Ranker tự động đọc đặc trưng từ tệp mô hình đã huấn luyện và tự động điền khuyết/bù đắp cột bị thiếu, ngăn ngừa hoàn toàn lỗi crash do bất đồng bộ cấu trúc đặc trưng giữa huấn luyện và suy luận.

