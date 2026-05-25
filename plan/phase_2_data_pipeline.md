# Phase 2: Data Pipeline & Point-in-time Feature Engineering

Mục tiêu: Chuyển đổi dữ liệu thô thành các đặc trưng chất lượng cao, tuyệt đối ngăn chặn rò rỉ dữ liệu tương lai (Data Leakage).

- [x] **1. Tiền xử lý dữ liệu (Preprocessing)**
  - [x] Sắp xếp toàn bộ dữ liệu thô theo cột `event_time`.
  - [x] Chia phiên (Sessionization): Gộp các hành vi của user theo cửa sổ thời gian (ví dụ: 30 phút).
- [x] **2. Khắc phục Dữ liệu khuyết thiếu (Missing Value) & Ngoại lai**
  - [x] Thay thế `fillna(0)` bằng giá trị định danh `<UNKNOWN>` (hoặc 'unknown_category', 'unknown_brand').
  - [x] Thực hiện Label Encoding hoặc Target Encoding cho các trường categorical.
  - [x] Áp dụng `Log Transform (log(x+1))` và chuẩn hóa Z-score cho cột `price`.
- [x] **3. Trọng số Phản hồi Ngầm (Implicit Feedback Weights)**
  - [x] Cập nhật `pseudo_label.py`: Gán trọng số tùy chỉnh thay vì nhị phân hóa (vd: `view` = 0.1, `cart` = 0.5, `purchase` = 1.0).
- [x] **4. Point-in-time Feature Engineering (Chống Time Travel)**
  - [x] Cập nhật `featurizer.py`.
  - [x] Tính `user_total_interactions`, `item_total_interactions` bằng phép cộng dồn (Cumulative Sum) tính đến thời điểm $t$. (Tuyệt đối KHÔNG dùng `groupby.size()` trên toàn tập).
  - [x] Bổ sung đặc trưng thời gian (Temporal features): `hour_of_day`, `day_of_week`.
  - [x] Bổ sung đặc trưng phiên point-in-time: `user_session_interaction_count` (số tương tác trong phiên hiện tại) và `item_session_popularity` (độ phổ biến sản phẩm theo số lượng phiên độc nhất).
- [x] **5. Time-based Splitting (Chia tập Train/Test theo thời gian)**
  - [x] Sắp xếp dữ liệu theo `event_time`.
  - [x] Chia tập Train (VD: 3 tuần đầu) và tập Test (VD: 1 tuần cuối). Không dùng `random_state` (chia ngẫu nhiên).