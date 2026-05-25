# Kế hoạch Triển khai: Hệ thống Đặc trưng Phiên & Triệu hồi Hai Kênh (Dual-Channel Recall)

Kế hoạch này chi tiết hóa cách thức bổ sung đặc trưng cấp phiên (session-level features) và cải tiến khâu Triệu hồi (Recall) thành cơ chế hai kênh: **Sở thích lâu dài** (Long-term) và **Nhu cầu phiên hiện tại** (Session-based).

---

## 1. Mục tiêu và Ý tưởng thiết kế

### 1.1. Bổ sung 2 Đặc trưng mới vào Featurizer
Để nâng cao độ nhạy của mô hình xếp hạng đối với các hành vi tức thời trong phiên:
1.  `user_session_interaction_count`: Số lượng tương tác của User trong phiên hiện tại tính đến thời điểm $t$. Đặc trưng này giúp mô hình nhận diện mức độ tích cực của người dùng trong phiên đó.
2.  `item_session_popularity`: Độ phổ biến của sản phẩm tính theo số lượng phiên độc nhất chứa sản phẩm này tính đến thời điểm $t$. Đặc trưng này giúp mô hình ưu tiên các sản phẩm đang có xu hướng "hot" theo phiên.

### 1.2. Triệu hồi Hai kênh (Dual-Channel Recall)
Thay vì chỉ triệu hồi dựa trên sở thích lâu dài của người dùng, chúng ta sẽ gộp kết quả từ 2 kênh:
*   **Kênh 1 (Long-term Preference):** Sử dụng Vector Embedding của User để truy vấn FAISS lấy các sản phẩm tương thích nhất dựa trên lịch sử lâu dài (như hiện tại).
*   **Kênh 2 (Current Session Needs):** Nếu User đang có phiên hoạt động tích cực (đã tương tác với một số sản phẩm trong vòng 30 phút qua):
    *   Lấy toàn bộ Vector Embeddings của các sản phẩm trong phiên hiện tại.
    *   Tính toán **Vector trung bình (Average Item Embedding)** đại diện cho mối quan tâm tức thì của phiên.
    *   Truy vấn FAISS bằng Vector trung bình này để tìm các sản phẩm tương tự nhất với các sản phẩm họ vừa xem.
*   **Gộp & Loại trùng (Merge & Deduplicate):** Ghép danh sách ứng viên từ cả hai kênh, loại bỏ trùng lặp và chuyển tiếp đến khâu xếp hạng.

---

## 2. Câu hỏi & Khảo sát ý kiến người dùng (Open Questions)

> [!IMPORTANT]
> Vui lòng cho ý kiến về các thiết kế chi tiết sau:
> 
> 1. **Cách tiếp cận Kênh 2 (Session-based Recall):** Phương án đề xuất là tính vector trung bình của các sản phẩm trong phiên rồi query FAISS (Option C). Phương án này tối ưu vì tận dụng được FAISS Index hiện tại mà không cần huấn luyện thêm mô hình mới. Bạn có đồng ý với phương án này không?
> 2. **Cách nạp dữ liệu phiên thời gian thực tại API Serving:** Trong môi trường thực tế, API cần biết các sản phẩm user vừa xem trong phiên hiện tại. Chúng tôi đề xuất cập nhật API FastAPI để nhận thêm một tham số tùy chọn `session_items: list[int]` (danh sách ID sản phẩm tương tác gần nhất trong phiên). Nếu không truyền vào, hệ thống sẽ tự động chỉ chạy Kênh 1. Bạn thấy phương án này có tiện lợi cho việc tích hợp và kiểm thử không?
> 3. **Định nghĩa `item_session_popularity`:** Định nghĩa đề xuất là số lượng phiên độc nhất (unique sessions) đã tương tác với sản phẩm này lũy kế đến thời điểm $t$. Điều này phản ánh chính xác độ phủ phiên của sản phẩm. Bạn có đồng ý không?

---

## 3. Các thay đổi đề xuất trong Mã nguồn (Proposed Changes)

### 3.1. Cấu hình hệ thống
#### [MODIFY] [config.py](file:///D:/programing/project/EDA_project/src/config.py)
*   Thêm `user_session_interaction_count` và `item_session_popularity` vào danh sách `NUMERICAL_FEATURES` trong cấu hình.
*   Bổ sung tham số cấu hình số lượng ứng viên triệu hồi cho từng kênh riêng biệt (ví dụ: `RECALL_TOP_K_LONG_TERM = 100`, `RECALL_TOP_K_SESSION = 100`).

---

### 3.2. Data Pipeline & Feature Engineering
#### [MODIFY] [featurizer.py](file:///D:/programing/project/EDA_project/src/data_pipeline/featurizer.py)
*   **Hàm `add_point_in_time_features()`:**
    *   Tính toán `user_session_interaction_count` lũy tiến point-in-time:
        ```python
        result["user_session_interaction_count"] = result.groupby(["user_id", "custom_session_id"]).cumcount()
        ```
    *   Tính toán `item_session_popularity` lũy tiến point-in-time:
        ```python
        first_item_session = ~result.duplicated(["product_id", "custom_session_id"])
        result["item_session_popularity"] = (
            first_item_session.groupby(result["product_id"]).cumsum()
            - first_item_session.astype(int)
        )
        ```
*   **Hàm `extract_item_features()`:**
    *   Tích hợp tính toán tổng số lượng phiên độc nhất lịch sử cho mỗi sản phẩm (`item_session_popularity` snapshot) để lưu vào Feature Store phục vụ suy luận:
        ```python
        item_sessions = df.groupby("product_id")["custom_session_id"].nunique().reset_index(name="item_session_popularity")
        item_features = item_features.merge(item_sessions, on="product_id", how="left")
        ```

---

### 3.3. Serving Pipeline & API
#### [MODIFY] [pipeline.py](file:///D:/programing/project/EDA_project/src/serving/pipeline.py)
*   Nâng cấp hàm `recommend()` nhận thêm tham số `session_items: Optional[list[int]] = None`.
*   Triển khai logic **Recall Hai Kênh**:
    1.  **Kênh 1:** Tìm ứng viên bằng User Embedding.
    2.  **Kênh 2:** Nếu `session_items` có dữ liệu, chuyển đổi các `product_id` thành index, lấy embeddings của chúng từ mô hình Recall, tính trung bình vector và truy vấn FAISS.
    3.  **Gộp ứng viên:** Ghép hai danh sách, loại bỏ sản phẩm trùng.
    4.  **Tạo Đặc trưng Chỉ thị:** Thêm các cột ảo `recalled_by_long_term` và `recalled_by_session` (0 hoặc 1) để cung cấp tín hiệu mạnh cho LightGBM Ranker.
*   Tính toán động đặc trưng `user_session_interaction_count` tại thời điểm gọi API: bằng độ dài của danh sách `session_items` truyền vào.

#### [MODIFY] [main_serve.py](file:///D:/programing/project/EDA_project/main_serve.py)
*   Cập nhật endpoint `/recommend/{user_id}` để chấp nhận nhận danh sách `session_items` thông qua Query Parameters (ví dụ: `/recommend/123?session_items=1001,1002`) hoặc chuyển endpoint thành `POST /recommend` nhận JSON body chứa `user_id` và `session_items`.

---

## 4. Kế hoạch Kiểm thử & Xác minh (Verification Plan)

### 4.1. Kiểm thử Tích hợp và Chạy Pipeline
1.  Chạy lại quy trình tiền xử lý và sinh đặc trưng:
    ```bash
    uv run python -m src.data_pipeline.run_pipeline
    ```
    Xác minh xem file `labeled_sessions.parquet` và `item_features.parquet` có chứa hai cột đặc trưng mới hay không.
2.  Chạy huấn luyện lại mô hình xếp hạng:
    ```bash
    uv run python -m src.ranking_model.run_ranking
    ```
    Kiểm tra xem mô hình LightGBM có học các đặc trưng mới thành công và đạt chỉ số NDCG/MRR tốt hơn hay không.

### 4.2. Kiểm thử API Phục vụ (Serving API Test)
Khởi động API Server:
```bash
uv run python main_serve.py
```
Gửi hai yêu cầu kiểm thử để đối chiếu:
1.  **Chỉ chạy Kênh 1 (Không có lịch sử phiên):**
    ```bash
    curl "http://127.0.0.1:8000/recommend/12345"
    ```
2.  **Chạy kết hợp Hai Kênh (Có lịch sử phiên):**
    ```bash
    curl "http://127.0.0.1:8000/recommend/12345?session_items=5700030,5700140"
    ```
    Kiểm tra xem API có hoạt động mượt mà, thời gian phản hồi có duy trì dưới 100ms và kết quả gợi ý có mang các sản phẩm tương tự với `session_items` lên trên hay không.
