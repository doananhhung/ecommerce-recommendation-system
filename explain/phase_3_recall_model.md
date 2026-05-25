# Phase 3: Giai Đoạn Triệu Hồi (Recall Model) & Bộ Lọc Hai Kênh (Dual-Channel Recall)

Khi hệ thống thương mại điện tử sở hữu hàng triệu sản phẩm, chúng ta không thể sử dụng các thuật toán học máy phức tạp để tính toán chi tiết cho từng sản phẩm đó đối với một khách hàng vì giới hạn về thời gian xử lý (độ trễ API phải <100ms). Giai đoạn Triệu hồi (Recall) ra đời để giải quyết bài toán này: nhanh chóng lọc thô, loại bỏ 99.9% sản phẩm không liên quan để giữ lại Top-K (ví dụ: 200) sản phẩm tiềm năng nhất từ nhiều kênh thông tin khác nhau.

---

## 🎯 Mục Tiêu Của Phase 3

1.  **Thu hẹp không gian tìm kiếm**: Lọc nhanh từ hàng triệu sản phẩm xuống còn 200 ứng viên sáng giá nhất cho mỗi khách hàng trong vài mili-giây.
2.  **Học biểu diễn ẩn (Embedding Learning)**: Biến các thực thể User và Item trừu tượng thành các chuỗi số (Vector Embedding 64 chiều) trong cùng một không gian hình học thông qua mô hình PyTorch.
3.  **Triệu hồi Hai kênh (Dual-Channel Recall)**: 
    *   *Kênh 1*: Khai thác sở thích lâu dài của người dùng dựa trên lịch sử tương tác tổng quan.
    *   *Kênh 2*: Khai thác nhu cầu tức thì, nóng hổi dựa trên các hành vi đang xảy ra tại phiên hiện tại.
4.  **Tối ưu hóa dữ liệu mất cân bằng**: Huấn luyện mô hình PyTorch với hàm mất mát **Focal Loss** kết hợp **Negative Sampling** để tập trung vào các hành vi chuyển đổi quan trọng (bỏ giỏ, mua hàng).
5.  **Đánh giá hiệu năng Triệu hồi**: Đo lường tỷ lệ bao phủ bằng chỉ số **Hit Rate@50 (Recall@50)**.

---

## 📁 Thư Mục Khởi Tạo & Vai Trò

Trong Phase này, mô hình học máy Deep Learning đầu tiên được phát triển bằng PyTorch trong thư mục `src/recall_model/` cùng với notebook thử nghiệm:

```
EDA_project/
├── models_store/
│   ├── recall_weights.pth        # Lưu trữ trọng số Embeddings của User và Item sau huấn luyện
│   └── item_embeddings.npy       # Ma trận nhúng 64 chiều của toàn bộ sản phẩm phục vụ FAISS Index
├── notebooks/
│   └── 03_Model_Exp.ipynb        # Vở bài tập chạy thử nghiệm và viết nháp mô hình PyTorch MF
└── src/
    └── recall_model/
        ├── __init__.py
        ├── model.py              # Định nghĩa mô hình Matrix Factorization (PyTorch)
        ├── dataset.py            # Lớp chuyển đổi Parquet dữ liệu phiên thành PyTorch Dataset
        ├── trainer.py            # Vòng lặp huấn luyện (Training Loop) tích hợp Focal Loss
        └── run_recall.py         # Kịch bản sinh mẫu âm, chạy huấn luyện và đánh giá Recall@50
```

### 🔹 Mục đích chi tiết của từng tệp:

*   **`notebooks/03_Model_Exp.ipynb`**:
    *   *Kỹ thuật*: Tệp Jupyter Notebook dùng để thử nghiệm nhanh (Prototyping) việc thiết kế mô hình PyTorch Matrix Factorization, xây dựng thử nghiệm lớp `Dataset` và `DataLoader` để nạp dữ liệu theo mini-batch, kiểm nghiệm tốc độ huấn luyện trên CPU/GPU, và xuất nháp ma trận nhúng của item trước khi chuyển đổi toàn bộ cấu trúc thành code module trong `src/recall_model/`.

*   **`model.py`**:
    *   *Kỹ thuật*: Định nghĩa lớp `MatrixFactorization` kế thừa `torch.nn.Module`. Lớp này khởi tạo hai bảng nhúng (`nn.Embedding`) cho User và Item với kích thước 64 chiều. Trọng số embeddings được khởi tạo bằng phương pháp **Xavier Uniform** giúp tăng tốc độ hội tụ. Hàm `forward` thực hiện tính Tích vô hướng (Dot Product) giữa User Embedding và Item Embedding rồi đưa qua hàm Sigmoid để ép giá trị dự đoán về khoảng `(0, 1)`.

## 🗺️ Thiết Kế Triệu Hồi Hai Kênh (Dual-Channel Recall Design)

Hệ thống gợi ý chuẩn công nghiệp của chúng ta sử dụng cơ chế Triệu hồi Song song hai kênh để bắt trọn cả hai khía cạnh sở thích của người dùng:

```mermaid
graph LR
    User[User ID & Session Items] --> Channel1[Kênh 1: Sở thích lâu dài]
    User --> Channel2[Kênh 2: Nhu cầu phiên hiện tại]
    
    Channel1 -->|User Embedding| FAISS1[FAISS KNN Search]
    Channel2 -->|Average Item Embedding| FAISS2[FAISS KNN Search]
    
    FAISS1 -->|Top 100 Candidates| Merge[Gộp & Loại trùng]
    FAISS2 -->|Top 100 Candidates| Merge
    
    Merge -->|Top 200 Candidates + Flags recalled_by_...| Ranker[LightGBM Ranking Stage]
```

### 1. Kênh 1 (Long-term Preference)
*   *Cơ chế*: Khi nhận yêu cầu `user_id`, hệ thống truy xuất Vector Embedding của User đó từ mô hình Matrix Factorization đã huấn luyện. Vector này đại diện cho sở thích lâu dài được tích lũy qua toàn bộ lịch sử tương tác.
*   *Tìm kiếm*: Sử dụng Vector User để truy vấn FAISS Index nhằm quét ra 100 sản phẩm lân cận có tích vô hướng (Dot Product) cao nhất.

### 2. Kênh 2 (Current Session Needs)
*   *Cơ chế*: Nếu người dùng đang thực hiện các hành động trong phiên hiện tại (ví dụ: đang click liên tiếp vào 3 mẫu điện thoại Samsung và 1 tai nghe trong 30 phút qua), nhu cầu tức thì của họ rất cao.
*   *Tính toán*: Hệ thống trích xuất Vector Item Embedding của toàn bộ các sản phẩm trong phiên hiện tại từ mô hình Recall, sau đó tính **Vector trung bình (Average Session Embedding)**. Vector trung bình này đại diện cho tọa độ bối cảnh hiện tại của phiên.
*   *Tìm kiếm*: Dùng Vector trung bình này để truy vấn FAISS Index nhằm quét ra 100 sản phẩm lân cận gần nhất. Phương án này tối ưu tuyệt đối vì nó tự động tìm ra các sản phẩm tương tự hoặc bổ trợ với các món đồ người dùng vừa xem trong phiên mà **không cần huấn luyện thêm bất cứ mô hình mới nào**.

### 3. Gộp ứng viên & Gán nhãn chỉ thị kênh (Candidate Merging & Tagging)
Hệ thống gộp danh sách ứng viên từ 2 kênh và thực hiện loại bỏ các sản phẩm trùng lặp. Đặc biệt, hệ thống gán thêm 2 đặc trưng chỉ thị:
*   `recalled_by_long_term`: Nhận giá trị `1.0` nếu sản phẩm được triệu hồi bởi Kênh 1, ngược lại là `0.0`.
*   `recalled_by_session`: Nhận giá trị `1.0` nếu sản phẩm được triệu hồi bởi Kênh 2, ngược lại là `0.0`.

Các chỉ thị này cung cấp tín hiệu cực kỳ mạnh mẽ để mô hình Xếp hạng (LightGBM Ranker) ở giai đoạn sau hiểu được "tại sao sản phẩm này lại được đề xuất" và đưa ra điểm số xếp hạng chính xác nhất.

---

## 🛠️ Công Nghệ & Khái Niệm Kỹ Thuật Sử Dụng

1.  **Vector Embedding (Không gian Nhúng)**:
    *   Biến một sản phẩm từ một ID vô nghĩa thành một chuỗi 64 số thực mã hóa các đặc tính ẩn của sản phẩm.
    *   *Phép nhân tích vô hướng (Dot Product)*: Nếu hai vector hướng cùng chiều, tích vô hướng của chúng sẽ rất cao -> User cực kỳ thích sản phẩm đó.
2.  **Focal Loss (Hàm mất mát tiêu điểm)**:
    *   $\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$
    *   Khi một mẫu dễ (như lượt View) đã được mô hình dự đoán chính xác với xác suất cao ($p_t \approx 1$), thừa số nhiễu $(1-p_t)^\gamma$ sẽ tiến về 0, làm giảm hẳn mức độ đóng góp của mẫu đó vào tổng lỗi. Mô hình dồn toàn bộ sức mạnh để học các mẫu khó (mua hàng, bỏ giỏ).
3.  **FAISS (Facebook AI Similarity Search)**:
    *   Thư viện tìm kiếm láng giềng gần nhất (ANN - Approximate Nearest Neighbors) siêu tốc được viết bằng C++. Nó cho phép tính toán khoảng cách vector giữa hàng nghìn User/Session và hàng triệu Item trong tích tắc (dưới 1ms), vượt xa hiệu năng tính toán thủ công bằng vòng lặp Python.
