# Phân tích Dữ liệu: Vấn đề và Giải pháp

## 1. Giá trị khuyết thiếu (Missing Values)

**Vấn đề:**
Tập dữ liệu ghi nhận tỷ lệ khuyết thiếu lớn tại `category_code` (31.8%) và `brand` (14.8%). Các giá trị này mang tính hệ thống (MAR/MNAR). Trong không gian vector n-chiều, việc thiếu hụt đặc trưng làm biến dạng khoảng cách hình học giữa các sản phẩm, ví dụ như khoảng cách Cosine:
$$Cosine(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

**Giải pháp:**
Không sử dụng các phương pháp gán nhược (imputation) cơ bản như mean/mode. Áp dụng kỹ thuật Embedding định danh riêng cho giá trị khuyết thiếu (tạo vector đại diện `<UNKNOWN>`). Phương pháp này đảm bảo cấu trúc topo của không gian nhúng không bị sai lệch.

## 2. Mất cân bằng dữ liệu cực đoan (Extreme Class Imbalance)

**Vấn đề:**
Tỷ lệ tương tác chênh lệch nghiêm trọng: `view` (1.936.686) so với `purchase` (33.877), tương đương tỷ lệ $\approx 57:1$.
Nếu sử dụng hàm mất mát Cross-Entropy tiêu chuẩn để phân loại:
$$CE(p_t) = -\log(p_t)$$
Tổng gradient trong quá trình tối ưu hóa mạng nơ-ron sẽ bị chi phối hoàn toàn bởi lớp đa số (`view`). Mô hình rơi vào cực tiểu cục bộ do luôn dự đoán `view` để tối thiểu hóa sai số.

**Giải pháp:**
Áp dụng Focal Loss để điều chỉnh biên độ Gradient động:
$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
Nhân tố điều chỉnh $(1 - p_t)^\gamma$ triệt tiêu gradient của các mẫu dễ nhận diện (chủ yếu là `view` khi $p_t \to 1$). Toàn bộ năng lực học tập của mạng nơ-ron được ép tập trung vào các mẫu thiểu số và khó nhận diện (`cart`, `purchase`).

## 3. Độ thưa thớt ma trận (User-Item Sparsity)

**Vấn đề:**
Ma trận User-Item có 295.883 users và 81.476 items nhưng chỉ có 2.000.000 tương tác. Độ thưa thớt đạt mức 99.9917%, xác định bằng công thức:
$$Sparsity = 1 - \frac{|Interactions|}{|Users| \times |Items|}$$
Thuật toán phân rã ma trận truyền thống ($R \approx P Q^T$) mất tính hiệu quả, dễ dẫn đến Overfitting do thiếu các điểm dữ liệu neo (anchor) để tối ưu hóa trọng số.

**Giải pháp:**
Chuyển đổi bài toán sang học máy dựa trên phản hồi ẩn (Implicit Feedback). Áp dụng thuật toán Bayesian Personalized Ranking (BPR) để tối ưu hóa thứ hạng dựa trên các cặp sản phẩm (pairwise learning) thay vì dự đoán giá trị tuyệt đối.

## 4. Phân phối ngoại lai của trường giá (Price Distribution)

**Vấn đề:**
Hàm mật độ xác suất của trường `price` lệch phải và có cấu trúc đuôi nặng (Mean = 296.9, Median = 161.9, Max = 2574). Việc đưa trực tiếp dữ liệu thô với các giá trị ngoại lai lớn vào mạng nơ-ron gây ra hiện tượng bùng nổ gradient (Exploding Gradients) và làm mất tính ổn định trọng số.

**Giải pháp:**
Xử lý co giãn phi tuyến tính để đưa dữ liệu về phân phối tiệm cận chuẩn. Sử dụng Log Transform:
$$x_{new} = \log(x + 1)$$
Tiếp tục áp dụng Standard Scaler (Z-score) cho $x_{new}$ nhằm chuẩn hóa dữ liệu về trung bình 0 và phương sai 1 trước khi đưa vào huấn luyện.