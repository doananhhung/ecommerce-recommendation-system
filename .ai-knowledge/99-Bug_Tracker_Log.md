# 9. Nhật ký Lỗi & Giải quyết (Bug Tracker Log)

File này được sử dụng để ghi chú lại toàn bộ các bug, lỗi tư duy hoặc vấn đề hệ thống gặp phải trong quá trình phát triển dự án. Việc ghi chép này giúp AI và lập trình viên dễ dàng tra cứu lại nguyên nhân (Root Cause) và cách khắc phục khi gặp các lỗi lặp lại.

---

## 📌 Template Ghi Bug (Mẫu)
Khi phát hiện lỗi, hãy copy mẫu này và thêm vào cuối danh sách:

```markdown
### [Bug ID] - Tên lỗi ngắn gọn
- **Ngày phát hiện:** DD/MM/YYYY
- **Thành phần:** (Ví dụ: PyTorch Model, FAISS, Data Pipeline...)
- **Mô tả lỗi:** (Traceback lỗi, hành vi không mong muốn, hiện tượng).
- **Căn nguyên (Root Cause):** (Tại sao lại xảy ra lỗi này?).
- **Giải pháp khắc phục:** (Cách sửa code, thay đổi config, lệnh terminal đã dùng).
- **Trạng thái:** [Open / Resolved]
```

---

## Danh sách Lỗi & Vấn đề (Issues Log)

*(Chưa có lỗi nào được ghi nhận. Hệ thống đang đợi bắt đầu triển khai code.)*

---

### [Bug 001] - NameError & Type Hint Warning Pylance trong Model Ranker
- **Ngày phát hiện:** 16/05/2026
- **Thành phần:** `src/serving/ranker.py` (Ranking Model / LightGBM)
- **Mô tả lỗi:** Khi gọi hàm `main_serve.py`, hệ thống báo lỗi `NameError: name 'np' is not defined` do việc sử dụng `np.ndarray` ở Type Hint nhưng chưa import thư viện `numpy`. Sau khi import `numpy`, Pylance tiếp tục báo lỗi `Type "tuple[ndarray | spmatrix | List[spmatrix], float]" is not assignable to return type "Tuple[ndarray, float]"`.
- **Căn nguyên (Root Cause):** Phương thức `predict` của `lgb.Booster` trong thư viện LightGBM trả về các kiểu dữ liệu khác nhau tuỳ thuộc vào đầu vào (có thể là Numpy Array, SciPy Sparse Matrix, hoặc List). Việc fix cứng Type Hint thành `Tuple[np.ndarray, float]` khiến trình kiểm tra kiểu (Type Checker) cảnh báo không tương thích.
- **Giải pháp khắc phục:** 
  1. Thêm dòng `import numpy as np`.
  2. Đổi Type Hint linh hoạt hơn bằng cách sử dụng `typing.Any` thay cho `np.ndarray`: Đổi thành `def predict(self, features_df: pd.DataFrame) -> Tuple[Any, float]:`.
- **Trạng thái:** [Resolved]

---

*(Khu vực này sẽ được mở rộng liên tục trong quá trình lập trình)*

---

## Checklist Vấn Đề Cần Fix Theo Thứ Tự Ưu Tiên

### P0 - Sửa Luồng Serving Và Mapping ID
- [x] **Persist encoder artifacts sau khi train recall**
  - **Thành phần:** `src/recall_model/run_recall.py`
  - **Vấn đề:** `LabelEncoder` cho `user_id` và `product_id` đang chỉ tồn tại trong RAM khi train, không được lưu ra disk.
  - **Tác động:** API chỉ dùng được `user_idx` nội bộ, không map ổn định từ user thật sang embedding index.
  - **Hướng fix:** Lưu `user_encoder`, `item_encoder`, và metadata `num_users`, `num_items`, `embedding_dim` vào `models_store/` hoặc `data/feature_store/`.

- [x] **Đổi API từ `user_idx` sang `user_id` thật**
  - **Thành phần:** `main_serve.py`, `src/serving/pipeline.py`
  - **Vấn đề:** Endpoint hiện tại `/recommend/{user_idx}` yêu cầu index đã encode, không phù hợp với dữ liệu/API thực tế.
  - **Tác động:** Client không biết `user_idx`; kết quả cũng khó diễn giải.
  - **Hướng fix:** Tạo endpoint `/recommend/{user_id}`; bên trong dùng encoder/mapping để chuyển sang `user_idx`.

- [x] **Trả về `product_id` thật thay vì chỉ `item_idx`**
  - **Thành phần:** `src/serving/pipeline.py`
  - **Vấn đề:** Response hiện trả `item_idx`, không phải mã sản phẩm gốc.
  - **Tác động:** Kết quả recommendation không dùng trực tiếp được ở UI/business layer.
  - **Hướng fix:** Dùng `item_encoder.inverse_transform()` hoặc bảng mapping `item_idx -> product_id`.

- [x] **Loại bỏ giả định `iloc` trong feature lookup**
  - **Thành phần:** `src/serving/pipeline.py`
  - **Vấn đề:** Code đang giả định `user_idx == row position` và `item_idx == row position` khi lookup feature bằng `iloc`.
  - **Tác động:** Sai feature nếu thứ tự DataFrame thay đổi hoặc encoder order khác với parquet order.
  - **Hướng fix:** Set index rõ ràng theo `user_id`/`product_id`, hoặc tạo feature store có khóa `user_idx`/`item_idx`.

### P1 - Sửa Data Leakage Và Chất Lượng Training
- [x] **Thay random split bằng time-based split**
  - **Thành phần:** `src/ranking_model/run_ranking.py`, data pipeline
  - **Vấn đề:** Ranking hiện dùng `train_test_split` random.
  - **Tác động:** Có nguy cơ data leakage, validation AUC/NDCG không phản ánh hiệu quả thực tế.
  - **Hướng fix:** Giữ `event_time` trong labeled data, chia train/validation/test theo mốc thời gian.

- [x] **Làm point-in-time features**
  - **Thành phần:** `src/data_pipeline/featurizer.py`
  - **Vấn đề:** Feature đang tính bằng `groupby` toàn bộ dữ liệu.
  - **Tác động:** Feature của quá khứ có thể nhìn thấy tương lai.
  - **Hướng fix:** Tính cumulative/rolling features theo `event_time`, chỉ dùng dữ liệu trước thời điểm dự đoán.

- [x] **Giữ thêm context trong labeled sessions**
  - **Thành phần:** `src/data_pipeline/pseudo_label.py`, `src/data_pipeline/run_pipeline.py`
  - **Vấn đề:** `labeled_sessions.parquet` hiện chỉ có `custom_session_id`, `product_id`, `user_id`, `label`.
  - **Tác động:** Khó split theo thời gian, khó đánh giá theo session/user, khó debug.
  - **Hướng fix:** Lưu thêm `event_time` đại diện, `event_type` mạnh nhất, `category_id`, `category_code`, `brand`, `price` nếu cần.

- [x] **Thêm sample weight cho implicit feedback**
  - **Thành phần:** `src/data_pipeline/pseudo_label.py`, `src/ranking_model/lgbm_train.py`
  - **Vấn đề:** Hiện `view=0`, `cart/purchase=1`; chưa biểu diễn mức độ tín hiệu khác nhau.
  - **Tác động:** Mất thông tin hành vi và khó xử lý imbalance.
  - **Hướng fix:** Tạo `sample_weight`, ví dụ `view=0.1`, `cart=0.5`, `purchase=1.0`, truyền vào LightGBM.

### P2 - Cải Thiện Recall Và Ranking
- [x] **Thêm negative sampling có kiểm soát cho recall**
  - **Thành phần:** `src/recall_model/dataset.py`, `src/recall_model/run_recall.py`
  - **Vấn đề:** Recall đang học trực tiếp từ pseudo-label hiện có, chưa có sampling chiến lược.
  - **Tác động:** Dễ bị imbalance và nhiễu từ view-only events chi phối.
  - **Hướng fix:** Sampling theo tỷ lệ, ví dụ 1 positive : 4 negatives, tránh lấy negative đã từng positive với user.

- [x] **Xem xét thay `BCELoss` bằng BPR/Focal Loss**
  - **Thành phần:** `src/recall_model/trainer.py`
  - **Vấn đề:** `BCELoss` đơn giản chưa tối ưu cho ranking/implicit feedback thưa.
  - **Tác động:** Recall candidates có thể kém chất lượng.
  - **Hướng fix:** Ưu tiên BPR Loss cho pairwise ranking; Focal Loss nếu vẫn giữ binary classification.

- [x] **Bù imbalance cho LightGBM**
  - **Thành phần:** `src/ranking_model/lgbm_train.py`
  - **Vấn đề:** LightGBM chưa dùng `is_unbalance`, `scale_pos_weight`, hoặc `sample_weight`.
  - **Tác động:** Model có thể thiên về class majority.
  - **Hướng fix:** Tính `scale_pos_weight = negative_count / positive_count` hoặc truyền sample weights.

- [x] **Đánh giá ranking theo group user/session**
  - **Thành phần:** `src/ranking_model/metrics.py`, `src/ranking_model/run_ranking.py`
  - **Vấn đề:** NDCG/MRR hiện tính trên toàn bộ test array, không theo từng user/session query group.
  - **Tác động:** Metric ranking có thể sai ý nghĩa nghiệp vụ.
  - **Hướng fix:** Tính NDCG@K/MRR@K theo từng `user_id` hoặc `custom_session_id`, rồi lấy trung bình.

### P3 - Maintainability, Config Và Tài Liệu
- [x] **Tạo cấu hình tập trung**
  - **Thành phần:** `src/config.py`, các script `run_*.py`, `main_train.py`, `main_serve.py`
  - **Vấn đề:** Path, batch size, embedding dim, epochs, top_k đang hard-code rải rác.
  - **Tác động:** Khó chạy lại trên dataset khác hoặc môi trường khác.
  - **Hướng fix:** Dùng config dataclass/env/CLI args; gom path và hyperparameters vào một nơi.

- [x] **Chuẩn hóa import để chạy được bằng module**
  - **Thành phần:** `src/data_pipeline/run_pipeline.py`, `src/recall_model/run_recall.py`, `src/ranking_model/run_ranking.py`
  - **Vấn đề:** Một số import dạng local như `from dataset import ...`, `from lgbm_train import ...`.
  - **Tác động:** Dễ lỗi khi chạy từ root bằng `python -m ...` hoặc khi package hóa.
  - **Hướng fix:** Dùng absolute import `from src.recall_model.dataset import ...` hoặc relative import nhất quán.

- [x] **Viết README hướng dẫn chạy end-to-end**
  - **Thành phần:** `README.md`
  - **Vấn đề:** README hiện trống.
  - **Tác động:** Người mới hoặc AI coding agent khó biết setup, train, serve, artifact cần có.
  - **Hướng fix:** Thêm mô tả project, data layout, lệnh `uv sync`, `uv run python main_train.py`, `uv run python main_serve.py`, endpoint mẫu.

- [x] **Thêm smoke tests hoặc script kiểm tra artifact**
  - **Thành phần:** `tests/` hoặc `scripts/`
  - **Vấn đề:** Chưa có kiểm thử tự động cho pipeline/artifact/serving.
  - **Tác động:** Dễ phát hiện lỗi muộn sau khi train lâu.
  - **Hướng fix:** Thêm test đọc parquet, load model, build FAISS, gọi `recommend()` với một user hợp lệ.
