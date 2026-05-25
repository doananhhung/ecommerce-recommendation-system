# Báo cáo Phân tích và Đối chiếu Đồng bộ: Codebase vs Kế hoạch Dự án

Tài liệu này cung cấp kết quả phân tích toàn diện cấu trúc codebase hiện tại của dự án **Hệ thống Gợi ý Hai Giai đoạn (Two-Stage Recommender System)** và đối chiếu với các bản kế hoạch chi tiết (Phases 1-5, Kế hoạch cải thiện dữ liệu) để kiểm tra mức độ đồng bộ và phát hiện các rủi ro, lỗi tiềm ẩn.

---

## I. Tổng quan Dự án & Cấu trúc Codebase Hiện tại

Hệ thống được thiết kế theo kiến trúc gợi ý hiện đại gồm **2 giai đoạn** cốt lõi:
1.  **Recall Stage (Matrix Factorization + PyTorch + FAISS):** Sàng lọc thô từ hàng triệu sản phẩm xuống 200 ứng viên tiềm năng bằng Vector Search.
2.  **Ranking Stage (LightGBM + Point-in-time Features):** Chấm điểm và xếp hạng chính xác Top 20 sản phẩm tối ưu dựa trên tương tác lịch sử và đặc trưng thời gian thực.

Codebase hiện tại được tổ chức rất quy củ:
-   `src/data_pipeline/`: Tiền xử lý dữ liệu, tạo session, tính toán đặc trưng point-in-time chống rò rỉ dữ liệu, và dán nhãn implicit feedback.
-   `src/recall_model/`: Định nghĩa dataset, mô hình Matrix Factorization, hàm huấn luyện với Focal Loss và tập lệnh chạy huấn luyện Recall.
-   `src/ranking_model/`: Trình quản lý huấn luyện LightGBM, các hàm đánh giá NDCG@K, MRR và tập lệnh huấn luyện Ranking.
-   `src/serving/`: Tìm kiếm vector bằng FAISS, chấm điểm bằng LightGBM và lớp Pipeline kết nối 2 giai đoạn phục vụ thực tế.
-   `main_train.py` & `main_serve.py`: Điểm kích hoạt end-to-end cho huấn luyện và API FastAPI phục vụ thời gian thực.

---

## II. Phân tích & Đối chiếu Đồng bộ theo từng Giai đoạn

Dưới đây là bảng đối chiếu chi tiết giữa **Kế hoạch đã đề ra** và **Code thực tế đang triển khai**:

### Phase 1: Phân tích Dữ liệu (EDA) & Thiết lập
| Mục tiêu trong Plan | Hiện trạng trong Code | Đánh giá đồng bộ |
| :--- | :--- | :--- |
| Khảo sát dữ liệu thô `2019-Oct.csv` (RAM-safe) | Đã triển khai đọc một phần dữ liệu thông qua tham số `PIPELINE_NROWS: 1000000` trong `config.py`. | **Hoàn toàn đồng bộ** |
| Nhận diện mất cân bằng lớp (96.8% `view`) | Có Notebook `01_EDA.ipynb` phân tích chi tiết. | **Hoàn toàn đồng bộ** |
| Phát hiện độ thưa thớt (99.99%) & khuyết thiếu | Có phân tích trong file lý thuyết và Notebook EDA. | **Hoàn toàn đồng bộ** |
| Nhận diện phân phối ngoại lai của `price` | Đã xác nhận trong Notebook EDA. | **Hoàn toàn đồng bộ** |

### Phase 2: Data Pipeline & Point-in-time Feature Engineering
| Mục tiêu trong Plan | Hiện trạng trong Code | Đánh giá đồng bộ |
| :--- | :--- | :--- |
| **Sessionization (30 phút)** | Viết trong `sessionizer.py`, tự động nhóm các hành vi cách nhau dưới 30 phút của cùng một user thành `custom_session_id`. | **Hoàn toàn đồng bộ** |
| **Log Transform Price** | Thực hiện trong `run_pipeline.py` (`np.log1p(price)`). | **Hoàn toàn đồng bộ** |
| **Implicit Feedback Weights** | Triển khai trong `pseudo_label.py`: `view = 0.1`, `cart = 0.5`, `purchase = 1.0`. Lấy tương tác mạnh nhất trong session. | **Hoàn toàn đồng bộ** |
| **Point-in-time Features (Không Leakage)** | Đã hiện thực hóa xuất sắc trong `featurizer.py` bằng các phép cộng dồn (`cumcount`, `cumsum` lũy tiến theo trục thời gian) thay vì `groupby` toàn tập. | **Hoàn toàn đồng bộ (Masterpiece!)** |
| **Time-based Splitting** | Triển khai trong `run_ranking.py` thông qua việc sort theo `event_time` và cắt 80/20 thay vì split random. | **Hoàn toàn đồng bộ** |

### Phase 3: Giai đoạn Triệu hồi (Recall Model)
| Mục tiêu trong Plan | Hiện trạng trong Code | Đánh giá đồng bộ |
| :--- | :--- | :--- |
| **Negative Sampling (tỷ lệ 1:4)** | Đã triển khai lấy mẫu âm tính ngẫu nhiên tại `run_recall.py` (Line 86-96) giúp cân bằng tập dữ liệu huấn luyện Recall. | **Hoàn toàn đồng bộ** |
| **Matrix Factorization (PyTorch)** | Triển khai mô hình Matrix Factorization với Xavier Uniform initialization trong `model.py`. | **Hoàn toàn đồng bộ** |
| **Focal Loss** | Đã viết `FocalLoss` tùy chỉnh trong `trainer.py` (alpha=0.25, gamma=2.0) giúp mô hình tập trung vào lớp thiểu số khó. | **Hoàn toàn đồng bộ** |
| **BPR Loss** (Tùy chọn nâng cao) | Chưa triển khai (Plan đang để trống `[ ]`). | **Khớp với kế hoạch** (Là tính năng mở rộng trong tương lai) |
| **Recall@50 Evaluation** | Đã triển khai tính toán Hit Rate@50 tự động bằng FAISS trong hàm `evaluate_recall` tại `run_recall.py`. | **Hoàn toàn đồng bộ** |

### Phase 4: Giai đoạn Xếp hạng (Ranking Model)
| Mục tiêu trong Plan | Hiện trạng trong Code | Đánh giá đồng bộ |
| :--- | :--- | :--- |
| **Cấu hình mất cân bằng LightGBM** | Bật tham số `"is_unbalance": True` trong `lgbm_train.py`. | **Hoàn toàn đồng bộ** |
| **Truyền Sample Weights vào LGBM** | Đã lấy trọng số từ Parquet và đưa vào `lgb.Dataset` (`train_weight` và `test_weight`). | **Hoàn toàn đồng bộ** |
| **Đánh giá bằng NDCG@K và MRR** | Triển khai các hàm toán học xếp hạng chuẩn trong `metrics.py` (có hỗ trợ group-based metrics cho từng user). | **Hoàn toàn đồng bộ** |
| **Lưu mô hình (`ranker_model.txt`)** | Đã lưu thành công tại `models_store/ranker_model.txt`. | **Hoàn toàn đồng bộ** |

---

## III. Phát hiện các điểm BẤT CẬP và LỖI NGHIÊM TRỌNG (Critical Bugs / Gaps)

Mặc dù phần lớn dự án được thiết kế cực kỳ bài bản và đồng bộ, quá trình đối chiếu code chi tiết đã phát hiện ra **3 điểm bất cập lớn**, trong đó có **1 lỗi thiết kế nghiêm trọng** sẽ khiến hệ thống phục vụ (Serving API) bị crash hoặc sai lệch kết quả khi triển khai thực tế:

### 1. Lỗi Nghiêm trọng: Bất đồng bộ đặc trưng giữa Huấn luyện và Suy luận (Feature Inconsistency Bug)
> [!CAUTION]
> **Hiện tượng:**
> -   **Khi huấn luyện (`src/ranking_model/run_ranking.py`):** Hệ thống tự động kiểm tra sự tồn tại của `category_code` và `brand` trong tập dữ liệu. Nếu có, nó sẽ thêm hai trường này vào danh sách đặc trưng phân loại (`features`) để huấn luyện LightGBM (tổng cộng **7 đặc trưng**):
>     ```python
>     features = ['user_total_interactions', 'user_total_sessions', 'item_total_interactions', 'item_unique_users', 'item_avg_price']
>     if 'category_code' in df.columns:
>         features.append('category_code')
>     if 'brand' in df.columns:
>         features.append('brand')
>     ```
> -   **Khi suy luận/serving (`src/serving/ranker.py`):** Hàm `predict` lại **hardcode cứng** chỉ lấy đúng **5 đặc trưng số**:
>     ```python
>     feature_cols = ['user_total_interactions', 'user_total_sessions', 'item_total_interactions', 'item_unique_users', 'item_avg_price']
>     X = features_df[feature_cols].copy()
>     ```
> 
> **Hậu quả:**
> Nếu dữ liệu thực tế đầu vào chứa `category_code` và `brand`, mô hình LightGBM sẽ được huấn luyện với **7 cột đầu vào**. Tuy nhiên, khi gọi API phục vụ, `ranker.py` chỉ truyền vào **5 cột**.
> LightGBM sẽ lập tiếp quăng lỗi crash: **"Number of features in data (5) doesn't match the number of features in model (7)"**, khiến API không thể trả về kết quả gợi ý.

### 2. Thiếu dữ liệu phân loại trong Serving Feature Store (Feature Store Gap)
> [!WARNING]
> Tại khâu lưu trữ đặc trưng cho Serving, file `src/data_pipeline/featurizer.py` trong hàm `extract_item_features()` chỉ nhóm và tính toán các chỉ số số học:
> ```python
> item_features = df.groupby("product_id").size().reset_index(name="item_total_interactions")
> # ... chỉ merge item_unique_users và item_avg_price
> ```
> Hàm này **hoàn toàn bỏ qua** trường `category_code` và `brand`! 
> Do đó, file `item_features.parquet` lưu trên đĩa cứng sẽ không có thông tin về danh mục và thương hiệu sản phẩm. Ngay cả khi chúng ta muốn sửa lỗi (1) bằng cách thêm `category_code` và `brand` vào Serving Pipeline, thì bảng tra cứu Serving Feature Store cũng không có dữ liệu để cung cấp, dẫn đến giá trị bị `NaN` (hoặc `<UNKNOWN>`).

### 3. Hardcoded và Phân tán Cấu hình Đặc trưng (Lack of Single Source of Truth)
> [!NOTE]
> Danh sách đặc trưng được sử dụng bởi mô hình xếp hạng hiện đang bị khai báo phân tán ở nhiều nơi (trong `run_ranking.py` và `ranker.py`) thay vì tập trung tại một nguồn duy nhất như `config.py`. Điều này làm tăng độ phức tạp khi bảo trì hệ thống và dễ dẫn đến sai sót khi muốn bổ sung hoặc bớt đi một đặc trưng nào đó.

---

## IV. Đề xuất Kế hoạch Khắc phục & Đồng bộ hóa Triệt để

Để khắc phục các điểm bất cập trên và đưa codebase về trạng thái đồng bộ 100% hoàn hảo, chúng tôi đề xuất thực hiện các bước nâng cấp sau:

### 1. Đồng bộ hóa Danh sách Đặc trưng trong `config.py`
Tập trung toàn bộ danh sách đặc trưng vào `ProjectConfig` để đảm bảo huấn luyện và suy luận luôn nhìn vào cùng một cấu hình:
```python
# Thêm vào src/config.py
@dataclass
class ProjectConfig:
    # ...
    # Features Config
    NUMERICAL_FEATURES: list = field(default_factory=lambda: [
        'user_total_interactions', 'user_total_sessions', 
        'item_total_interactions', 'item_unique_users', 'item_avg_price'
    ])
    CATEGORICAL_FEATURES: list = field(default_factory=lambda: [
        'category_code', 'brand'
    ])
```

### 2. Cập nhật `extract_item_features` để lưu trữ Categorical Data
Sửa đổi hàm `extract_item_features` trong `src/data_pipeline/featurizer.py` để trích xuất và lưu kèm các trường danh mục/thương hiệu mới nhất của sản phẩm:
```python
def extract_item_features(df: pd.DataFrame) -> pd.DataFrame:
    # Lấy thông tin tương tác số học như cũ...
    item_features = df.groupby("product_id").size().reset_index(name="item_total_interactions")
    # ...
    
    # Trích xuất category_code và brand mới nhất của từng sản phẩm (tránh null nếu có thể)
    if 'category_code' in df.columns or 'brand' in df.columns:
        cols_to_keep = [col for col in ['product_id', 'category_code', 'brand'] if col in df.columns]
        static_info = df.sort_values('event_time').drop_duplicates('product_id', keep='last')[cols_to_keep]
        item_features = item_features.merge(static_info, on='product_id', how='left')
        
    return item_features
```

### 3. Cập nhật `src/serving/ranker.py` để sử dụng đầy đủ Đặc trưng
Cập nhật `LightGBMRanker.predict()` để lấy động danh sách đặc trưng từ `config` hoặc tự động nhận diện từ cấu trúc mô hình đã load:
```python
    def predict(self, features_df: pd.DataFrame) -> Tuple[Any, float]:
        if self.model is None:
            raise ValueError("Model is not loaded. Call load_model() first.")
            
        # Lấy trực tiếp danh sách đặc trưng mà mô hình đã học khi train
        feature_cols = self.model.feature_name()
        
        X = features_df.copy()
        
        # Điền khuyết tương tự như lúc train
        if 'category_code' in X.columns:
            X['category_code'] = X['category_code'].fillna('<UNKNOWN>').astype('category')
        if 'brand' in X.columns:
            X['brand'] = X['brand'].fillna('<UNKNOWN>').astype('category')
            
        numeric_cols = X.select_dtypes(include=['number']).columns
        X[numeric_cols] = X[numeric_cols].fillna(0)
        
        X = X[feature_cols] # Đảm bảo đúng thứ tự và đủ số lượng đặc trưng
        
        start_time = time.time()
        preds = self.model.predict(X)
        elapsed = time.time() - start_time
        
        return preds, elapsed
```

---

## V. Kết luận

Dự án hiện tại có chất lượng codebase **rất cao**, triển khai toán học chính xác (đặc biệt là việc xử lý **Point-in-Time Features** và **Focal Loss** cực kỳ chuẩn mực). 

Tuy nhiên, sự bất đồng bộ về đặc trưng giữa hai khâu **Huấn luyện** (sử dụng thêm Categorical) và **Suy luận** (chỉ sử dụng Numerical) là một lỗ hổng nghiêm trọng cần được vá trước khi đem hệ thống đi vận hành thực tế. Áp dụng các đề xuất tại mục IV sẽ giúp dự án đạt trạng thái **Hoàn toàn đồng bộ & Sẵn sàng sản xuất (Production-Ready)**.
