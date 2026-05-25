# Báo Cáo Giải Thích Thư Mục Notebooks & Đối Chiếu Đồng Bộ Hóa Dữ Liệu Tài Liệu (Explain Folder Audit)

Tài liệu này cung cấp:
1.  **Hướng dẫn chi tiết về vai trò của từng tệp Jupyter Notebook** trong thư mục `notebooks/`.
2.  **Kết quả đối chiếu (Audit)** giữa tài liệu giải thích kỹ thuật trong thư mục `explain/` và mã nguồn triển khai thực tế trong thư mục `src/`, nhằm đánh giá mức độ đồng bộ và chỉ ra các khoảng trống thông tin.

---

## 📓 I. Vai Trò Chi Tiết Của Các Tệp Trong Thư Mục `notebooks/`

Thư mục `notebooks/` chứa các vở bài tập Jupyter Notebook tương tác. Đây là nơi các kỹ sư phát triển thử nghiệm giải pháp (Prototyping), phân tích trực quan hóa (Visualization) và kiểm nghiệm toán học trước khi đóng gói thành các mô-đun code Python tối ưu trong thư mục `src/`.

Dưới đây là chi tiết vai trò của 4 notebooks theo đúng tiến trình phát triển:

### 1. [01_EDA.ipynb](file:///D:/programing/project/EDA_project/notebooks/01_EDA.ipynb)
*   **Giai đoạn tương ứng:** Phase 1: Exploratory Data Analysis & Setup
*   **Nhiệm vụ chính:**
    *   Đọc thử nghiệm 1 triệu dòng đầu tiên từ tập dữ liệu khổng lồ `2019-Oct.csv` để đảm bảo an toàn bộ nhớ RAM.
    *   Tính toán chỉ số thưa thớt (**Sparsity**) của ma trận tương tác khách hàng - sản phẩm (phát hiện độ thưa đạt tới **99.99%**).
    *   Phân tích sự phân phối của hành vi (`event_type`), nhận diện mất cân bằng lớp cực đoan (**View chiếm 96.8%**, **Cart chiếm 2%**, **Purchase chiếm 1.2%**).
    *   Trực quan hóa phân phối giá cả sản phẩm (`price`), phát hiện phân phối lệch phải (skewed) rất dài và xác nhận sự cần thiết của phép biến đổi Log-Transform.
*   **Mô-đun thực tế được đóng gói:** Tạo tiền đề thiết kế cấu hình ProjectConfig trong `src/config.py`.

### 2. [02_Feature_Eng.ipynb](file:///D:/programing/project/EDA_project/notebooks/02_Feature_Eng.ipynb)
*   **Giai đoạn tương ứng:** Phase 2: Data Pipeline & Point-in-time Feature Engineering
*   **Nhiệm vụ chính:**
    *   **Thử nghiệm Sessionization:** Viết nháp thuật toán tính khoảng cách thời gian giữa các tương tác liên tiếp của từng User để chia phiên mua sắm với ngưỡng 30 phút.
    *   **Thử nghiệm Pseudo-labeling:** Xây dựng logic gán trọng số hành vi implicit (`view=0.1`, `cart=0.5`, `purchase=1.0`) và chọn tương tác mạnh nhất trong phiên.
    *   **Thử nghiệm Đặc trưng Point-in-time:** Viết nháp các hàm tính toán đặc trưng tích lũy lũy tiến không gây rò rỉ dữ liệu tương lai (No-Lookahead Leakage).
*   **Mô-đun thực tế được đóng gói:** Đóng gói thành `sessionizer.py`, `pseudo_label.py`, `featurizer.py` và `run_pipeline.py` trong thư mục `src/data_pipeline/`.

### 3. [03_Model_Exp.ipynb](file:///D:/programing/project/EDA_project/notebooks/03_Model_Exp.ipynb)
*   **Giai đoạn tương ứng:** Phase 3: Recall Model Prototyping
*   **Nhiệm vụ chính:**
    *   Mã hóa thử nghiệm chuỗi User ID và Product ID khổng lồ thành các index số nguyên liên tục bắt đầu từ 0 thông qua `LabelEncoder` của Scikit-learn.
    *   Định nghĩa lớp `ImplicitFeedbackDataset` kế thừa từ `torch.utils.data.Dataset` và cấu hình `DataLoader` để tải mini-batch.
    *   Xây dựng cấu trúc mô hình mạng Matrix Factorization cơ bản trong PyTorch với lớp nhúng Embedding layer.
    *   Kiểm nghiệm tốc độ huấn luyện trên CPU/GPU và xuất trọng số embeddings tĩnh của sản phẩm để chuẩn bị cho việc nạp FAISS.
*   **Mô-đun thực tế được đóng gói:** Đóng gói thành `model.py`, `dataset.py`, `trainer.py` và `run_recall.py` trong thư mục `src/recall_model/`.

### 4. [04_Ranking_Exp.ipynb](file:///D:/programing/project/EDA_project/notebooks/04_Ranking_Exp.ipynb)
*   **Giai đoạn tương ứng:** Phase 4: Ranking Model Prototyping (LightGBM)
*   **Nhiệm vụ chính:**
    *   Đọc và ghép nối (Join) tập dữ liệu tương tác Parquet với các đặc trưng snapshot của User/Item lấy từ Feature Store.
    *   Thử nghiệm thuật toán huấn luyện Gradient Boosting của thư viện **LightGBM** dạng phân loại nhị phân (Binary Classification).
    *   Thử nghiệm truyền `sample_weight` trực tiếp vào hàm huấn luyện của LightGBM để nhấn mạnh hành vi mua hàng.
    *   Đo lường độ chính xác xếp hạng bằng NDCG và MRR trên tập kiểm thử tĩnh.
*   **Mô-đun thực tế được đóng gói:** Đóng gói thành `lgbm_train.py`, `metrics.py` và `run_ranking.py` trong thư mục `src/ranking_model/`.

---

## 🔍 II. Kết Quả Đối Chiếu Đồng Bộ Hóa Tài Liệu Giải Thích (Explain Audit)

Tôi đã đối chiếu toàn bộ nội dung trong thư mục tài liệu `explain/` với cấu trúc mã nguồn Python thực tế trong thư mục `src/`. Kết quả thu được như sau:

### 1. Điểm Khuyết Thiếu (Khoảng Trống Tài Liệu)
*   **Thiếu giải thích về thư mục `notebooks/`:** 
    *   Ngoại trừ file [phase_1_eda_and_setup.md](file:///D:/programing/project/EDA_project/explain/phase_1_eda_and_setup.md) có nhắc đến tệp `01_EDA.ipynb`, các tài liệu giải thích của Phase 2, Phase 3 và Phase 4 hoàn toàn **bỏ qua** việc giải thích vai trò của các tệp `02_Feature_Eng.ipynb`, `03_Model_Exp.ipynb` và `04_Ranking_Exp.ipynb`. Điều này gây khó khăn cho các lập trình viên mới khi muốn tìm hiểu quy trình thử nghiệm.
*   **Giải pháp:** Tôi đã bổ sung toàn bộ lời giải thích chi tiết về vai trò của các notebooks này tại mục I của báo cáo này và đề xuất cập nhật thêm liên kết trong tệp `explain/overview.md`.

---

### 2. Mức Độ Đồng Bộ Hóa Về Kỹ Thuật (Code vs Explain Sync)
Nhìn chung, nội dung lý thuyết kỹ thuật trong `explain/` đạt mức độ **đồng bộ rất cao (gần như 100%)** với mã nguồn thực tế nhờ các đợt cập nhật và cải tiến hệ thống gần đây:

#### A. Đồng bộ hóa Kỹ thuật Point-in-time (Phase 2):
*   *Lý thuyết giải thích:* Tài liệu [phase_2_data_pipeline.md](file:///D:/programing/project/EDA_project/explain/phase_2_data_pipeline.md) giải thích rất kỹ cơ chế tránh Data Leakage bằng phép cộng dồn lũy tiến `.cumcount()` và `.cumsum()`.
*   *Mã nguồn thực tế:* File `src/data_pipeline/featurizer.py` được triển khai chuẩn mực bằng các phép toán tương tự trên Pandas, không có bất kỳ sai lệch nào.

#### B. Đồng bộ hóa Triệu hồi Hai Kênh & Focal Loss (Phase 3):
*   *Lý thuyết giải thích:* Tài liệu [phase_3_recall_model.md](file:///D:/programing/project/EDA_project/explain/phase_3_recall_model.md) mô tả chi tiết sơ đồ triệu hồi 2 kênh: Kênh 1 dựa trên User Embedding, Kênh 2 dựa trên Average Session Embedding từ các sản phẩm đang xem, gộp lại thông qua FAISS Index FlatIP. Đồng thời, giải thích công thức Focal Loss với $\alpha=0.25$ và $\gamma=2.0$.
*   *Mã nguồn thực tế:* 
    *   `src/serving/pipeline.py` cài đặt chính xác logic trung bình cộng vector của phiên và truy vấn FAISS song song.
    *   `src/recall_model/trainer.py` dòng 43 định nghĩa đúng lớp `FocalLoss(alpha=0.25, gamma=2.0)`.

#### C. Đồng bộ hóa Cấu hình Đặc trưng Động & Trình Xếp Hạng Kháng Lỗi (Phase 4 & 5):
*   *Lý thuyết giải thích:* Tài liệu mô tả cơ chế LightGBM tự động đọc đặc trưng động bằng thuộc tính `self.model.feature_name()`, tự động bù đắp các đặc trưng bị khuyết (NaN hoặc `<UNKNOWN>`) và gán nhãn chỉ thị nguồn gốc (`recalled_by_long_term`, `recalled_by_session`).
*   *Mã nguồn thực tế:* 
    *   `src/serving/ranker.py` tự động đọc cấu trúc cây quyết định từ tệp `.txt` đã lưu, so khớp, điền khuyết động cho categorical và numerical cực kỳ an toàn.
    *   `src/serving/pipeline.py` tính toán động độ dài của `session_items` làm giá trị đặc trưng `user_session_interaction_count` tại thời điểm thực tế, gộp các nhãn chỉ thị kênh và sắp xếp điểm số chính xác.

---

## 💡 III. Kết Luận & Hành Động Tiếp Theo
*   Mã nguồn (`src/`) và tài liệu giải thích kỹ thuật (`explain/`) hiện đang ở trạng thái **đồng bộ hoàn hảo về mặt logic thuật toán và cấu hình đặc trưng**.
*   Để giải quyết triệt để vấn đề thiếu sót giải thích thư mục `notebooks/`, tệp báo cáo này (`artifact/notebooks_and_codebase_sync.md`) đã được lưu trữ làm nguồn tham chiếu chính thức cho toàn bộ luồng Prototyping của dự án.
