# Kế hoạch cải thiện Dữ liệu (Data Improvement Plan)

Dựa trên các phân tích từ quá trình EDA (`notebooks/01_EDA.ipynb`), chúng ta đã phát hiện ra một số vấn đề nghiêm trọng về dữ liệu có thể làm giảm hiệu suất của mô hình. Dưới đây là kế hoạch chi tiết để khắc phục các vấn đề này.

## 1. Vấn đề Mất cân bằng Lớp (Class Imbalance) nghiêm trọng
**Phân tích từ EDA:**
- `view` (Xem): ~1.93 triệu lượt (chiếm ~96.8%)
- `purchase` (Mua hàng): ~33.8 nghìn lượt (chiếm ~1.7%)
- `cart` (Thêm vào giỏ): ~29.4 nghìn lượt (chiếm ~1.5%)

Hiện tại, lớp negative (`view`) đang áp đảo hoàn toàn lớp positive (`cart`, `purchase`). Tỷ lệ xấp xỉ 1:30. Mô hình sẽ có xu hướng dự đoán mọi thứ là 0 (negative) để đạt độ chính xác (accuracy) cao nhưng lại bỏ lỡ các dự đoán mua hàng thực sự (recall cực thấp).

### Giải pháp:
**A. Tại khâu Huấn luyện Mô hình Recall (PyTorch):**
1.  **Negative Sampling (Lấy mẫu âm tính):** Trong class `ImplicitFeedbackDataset` (`src/recall_model/dataset.py`), thay vì lấy toàn bộ 1.93 triệu dòng `view`, chúng ta nên lấy mẫu ngẫu nhiên (sample) số lượng `view` theo tỷ lệ nhất định so với số lượng `purchase`/`cart` (ví dụ: 1 positive : 4 negatives). Điều này giúp giảm nhiễu và tăng tốc độ huấn luyện.
2.  **Chỉnh sửa Loss Function:** Thêm tham số `pos_weight` vào hàm `BCELoss` (hoặc chuyển sang dùng `BCEWithLogitsLoss`) trong file `src/recall_model/trainer.py` để phạt nặng hơn khi mô hình dự đoán sai lớp thiểu số (positive).

**B. Tại khâu Huấn luyện Mô hình Ranking (LightGBM):**
1.  **Sử dụng cấu hình cân bằng của LightGBM:** Trong file `src/ranking_model/lgbm_train.py`, cấu hình `params` cần được bổ sung tham số `is_unbalance: True` (hoặc tính toán thủ công và gán vào `scale_pos_weight`). LightGBM sẽ tự động điều chỉnh trọng số của các lớp để bù đắp cho sự mất cân bằng.

---

## 2. Vấn đề Dữ liệu Khuyết thiếu (Missing Values)
**Phân tích từ EDA:**
- Cột `category_code`: Bị thiếu (null) 31.8%
- Cột `brand`: Bị thiếu (null) 14.8%

Hiện tại, hệ thống (tại `run_ranking.py` và `ranker.py`) đang giải quyết bằng cách gọi thẳng `df.fillna(0)`. Việc điền số `0` cho một đặc trưng dạng chuỗi hoặc categorical (phân loại) là không hợp lý và làm mất thông tin ngữ nghĩa.

### Giải pháp:
1.  **Chiến lược điền khuyết (Imputation) đúng chuẩn:** 
    - Với `brand`: Thay vì điền `0`, hãy điền chuỗi `'unknown_brand'`.
    - Với `category_code`: Thay vì điền `0`, hãy điền chuỗi `'unknown_category'`. Có thể kết hợp `category_id` để nội suy nếu có ID đó nhưng bị thiếu tên code.
2.  **Mã hóa Đặc trưng (Feature Encoding):** Cần tích hợp Label Encoding hoặc Target Encoding vào trong `src/data_pipeline/featurizer.py` để mô hình LightGBM hiểu được các cột string này, đồng thời tận dụng được các danh mục bị thiếu dưới dạng một nhóm chung ('unknown').

---

## 3. Khai thác thêm tín hiệu Ngầm (Implicit Feedback) & Thời gian
**Phân tích từ EDA:**
- Hàm `pseudo_label.py` hiện gán: `view=0`, `cart=1`, `purchase=1`. Tuy nhiên, cường độ của 'mua hàng' chắc chắn phải cao hơn 'thêm vào giỏ'.
- Thời gian `event_time` đang bị bỏ phí.

### Giải pháp:
1.  **Trọng số tùy chỉnh (Custom Sample Weights):** Sửa lại logic trong `pseudo_label.py`. Thay vì gán nhãn cứng 0 và 1, hãy truyền trọng số vào mô hình. Ví dụ: `view` = 0.1, `cart` = 0.5, `purchase` = 1.0. Các framework như LightGBM hỗ trợ rất tốt `sample_weight`.
2.  **Trích xuất đặc trưng thời gian (Temporal Features):** Bổ sung vào `featurizer.py` các đặc trưng như:
    - Mua vào giờ nào trong ngày (`hour_of_day`)
    - Mua vào ngày nào trong tuần (`day_of_week`)
    - Thời gian tương tác cuối cùng (recency)
    Các yếu tố này thường ảnh hưởng rất lớn đến quyết định mua hàng trong thực tế.

---

## 4. LỖI NGHIÊM TRỌNG: Rò rỉ dữ liệu tương lai (Data Leakage / Time Travel)
**Phân tích từ Codebase (Bỏ sót từ bản kế hoạch ban đầu):**
Hệ thống hiện tại đang vi phạm nguyên tắc số 1 của Recommender System liên quan đến `Timestamp`:
1. **Chia tập Train/Test sai cách:** Trong `src/ranking_model/run_ranking.py`, hệ thống đang sử dụng `train_test_split(X, y, test_size=0.2, random_state=42)`. Việc lấy mẫu ngẫu nhiên (random split) đã trộn lẫn dữ liệu quá khứ và tương lai. Mô hình đang dùng thông tin tương lai để dự đoán quá khứ.
2. **Tính toán Feature bị Leakage:** Trong `src/data_pipeline/featurizer.py`, các biến như `user_total_interactions` được `groupby` và `size()` trên TOÀN BỘ tập dữ liệu. Nghĩa là tại thời điểm $t$, mô hình đã biết được tổng số lượt tương tác của User trong cả tương lai ($t+1, t+2$). Khi mang ra chạy thực tế (Serving), kết quả sẽ thảm họa vì dữ liệu tương lai chưa xảy ra.

### Giải pháp CHÍNH VÀ CẤP BÁCH:
1.  **Sửa thuật toán chia Train/Test:** BẮT BUỘC phải chia dữ liệu theo Trình tự thời gian (Time-based Splitting). Ví dụ: Lấy dữ liệu 3 tuần đầu làm Train, 1 tuần cuối làm Test (Dựa trên `event_time`), không bao giờ được dùng random seed.
2.  **Sửa thuật toán tạo Đặc trưng (Point-in-time Features):** `featurizer.py` phải được viết lại để tính toán Feature dựa trên Cửa sổ thời gian trượt (Rolling Window) hoặc tích lũy tới thời điểm hiện tại (Cumulative sum up to $t$), tuyệt đối không được phép Groupby gộp cả quá khứ và tương lai.