# Phase 1: Phân Tích Dữ Liệu Khám Phá (EDA) & Thiết Lập Hệ Thống

Phase 1 là bước đi nền móng đầu tiên của dự án. Trước khi xây dựng bất kỳ mô hình AI phức tạp nào, chúng ta cần chuẩn bị môi trường lập trình chuẩn hóa và tiến hành "thăm khám" dữ liệu thô để thấu hiểu hành vi của khách hàng, nhận diện các điểm nghẽn kỹ thuật và đề xuất giải pháp xử lý phù hợp.

---

## 🎯 Mục Tiêu Của Phase 1

1. **Chuẩn hóa môi trường phát triển**: Đảm bảo tất cả các nhà phát triển đều chạy chung một phiên bản thư viện, tránh lỗi xung đột hệ thống ("Mã chạy trên máy tôi nhưng lỗi trên máy bạn").
2. **Thấu hiểu dữ liệu thô (EDA)**: Nhận diện phân phối của các hành vi tương tác, tính toán độ thưa thớt của dữ liệu, phân tích đặc trưng giá cả, và xử lý dữ liệu khuyết thiếu.
3. **Định hình thiết kế mô hình**: Dựa trên kết quả thống kê thực tế để lựa chọn cấu trúc thuật toán tối ưu cho các Phase sau.

---

## 📁 Thư Mục Khởi Tạo & Vai Trò

Trong Phase này, cấu trúc dự án bắt đầu được định hình với các thư mục và tệp cốt lõi sau:

```
EDA_project/
├── .python-version          # Chỉ định phiên bản Python chuẩn hóa (Python 3.12)
├── pyproject.toml           # Định nghĩa cấu hình dự án và danh sách thư viện cần cài đặt
├── uv.lock                  # Khóa chính xác phiên bản của mọi package phụ thuộc
├── data/
│   └── raw/
│       └── 2019-Oct.csv     # File dữ liệu thô chứa lịch sử tương tác thương mại điện tử
└── notebooks/
    └── 01_EDA.ipynb         # Vở bài tập phân tích dữ liệu khám phá trực quan
```

### 🔹 Mục đích chi tiết của từng tệp:

*   **`pyproject.toml` & `uv.lock`**:
    *   *Kỹ thuật*: Thay vì sử dụng `requirements.txt` truyền thống dễ bị cài sai phiên bản hoặc giải quyết phụ thuộc chậm, dự án sử dụng công cụ quản lý package thế hệ mới **`uv`**. File `pyproject.toml` khai báo các thư viện lớn như `torch`, `lightgbm`, `faiss-cpu`, `fastapi`, `pandas`. File `uv.lock` đóng vai trò lưu lại "ảnh chụp" chính xác của cây thư viện phụ thuộc để tái tạo môi trường lập trình đồng nhất 100%.
*   **`data/raw/2019-Oct.csv`**:
    *   *Kỹ thuật*: Tệp dữ liệu hành vi người dùng cực lớn từ trang thương mại điện tử thực tế. Mỗi bản ghi (dòng) ghi lại một hành vi tương tác bao gồm: thời gian (`event_time`), loại hành vi (`event_type`: view, cart, purchase), mã sản phẩm (`product_id`), mã danh mục (`category_code`), thương hiệu (`brand`), giá (`price`), mã khách hàng (`user_id`), và mã phiên (`user_session`).
*   **`notebooks/01_EDA.ipynb`**:
    *   *Kỹ thuật*: File Jupyter Notebook tương tác dùng để chạy code phân tích nhanh, vẽ đồ thị. Nơi đây thực hiện đọc thử nghiệm một phần dữ liệu (ví dụ: 1 triệu dòng đầu tiên) để tránh tràn bộ nhớ RAM, sau đó tính toán các chỉ số thống kê cốt lõi.

---

## 🛠️ Công Nghệ & Khái Niệm Kỹ Thuật Sử Dụng

1.  **Quản lý môi trường với `uv`**: Công cụ được phát triển bằng ngôn ngữ Rust giúp cài đặt thư viện nhanh gấp 10-100 lần so với `pip` và `poetry`.
2.  **Pandas & Numpy**: Công cụ thao tác dữ liệu dạng bảng và mảng số học hiệu năng cao trên RAM.
3.  **Class Imbalance (Mất cân bằng dữ liệu cực đoan)**: 
    *   *Thực tế*: Phân tích số lượng `event_type` phát hiện: **View chiếm ~96.8%**, **Cart chiếm ~2.0%**, **Purchase chiếm ~1.2%**.
    *   *Hệ quả*: Nếu huấn luyện mô hình dự đoán nhị phân thông thường, mô hình chỉ cần luôn đoán là "User chỉ xem hàng (view)" thì độ chính xác đã đạt tới 96.8%. Do đó, ta bắt buộc phải sử dụng **Focal Loss (Phase 3)** hoặc **Sample Weights (Phase 4)** để phạt nặng mô hình nếu đoán sai hành vi mua hàng.
4.  **Matrix Sparsity (Độ thưa thớt của ma trận tương tác)**:
    *   *Công thức*: $\text{Sparsity} = 1 - \frac{\text{Số lượng tương tác thực tế}}{\text{Tổng số User} \times \text{Tổng số Item}}$
    *   *Thực tế*: Kết quả EDA cho thấy độ thưa thớt đạt tới **99.99%**. Tức là trung bình một người dùng chỉ tương tác với một phần vạn số sản phẩm có trong hệ thống. Điều này khiến cho các thuật toán so khớp dựa trên luật thông thường bị bất khả thi, đòi hỏi phải sử dụng **Matrix Factorization (Phase 3)** để ánh xạ họ vào không gian vector ẩn.
5.  **Skewed Price Distribution (Phân phối lệch phải của giá cả)**:
    *   *Thực tế*: Đa số sản phẩm có giá rẻ, chỉ có một số ít sản phẩm có giá cực kỳ đắt (điện thoại thông minh, trang sức). Đồ thị phân phối giá có đuôi rất dài về bên phải.
    *   *Hệ quả*: Để thuật toán học máy không bị nhiễu bởi các sản phẩm giá cực khủng, ta cần áp dụng phép biến đổi **Log Transform: $\log(x + 1)$ (Phase 2)** nhằm kéo phân phối giá về dạng chuẩn (hình chuông).

---

## 💡 Hình Dung Trực Quan (Ví Dụ Dễ Hiểu)

> **Ví dụ về "Người khảo sát siêu thị khổng lồ"**
> 
> Hãy tưởng tượng bạn được thuê về làm giám đốc vận hành cho một đại siêu thị có diện tích bằng 100 sân bóng đá, chứa **1 triệu mặt hàng** khác nhau và đón tiếp **100.000 khách hàng** mỗi ngày. Bạn không thể lập tức thiết kế một Robot hướng dẫn mua sắm nếu chưa biết gì về siêu thị này.
> 
> **Phase 1** chính là việc bạn cầm một cuốn sổ tay (`01_EDA.ipynb`) đi khảo sát thực địa siêu thị:
> *   Bạn nhận ra: Cứ 100 người vào siêu thị thì có tới 97 người chỉ đi lướt qua nhìn đồ rồi đi tiếp (View), chỉ có 2 người nhặt đồ bỏ vào giỏ (Cart), và duy nhất 1 người thực sự ra quầy thanh toán (Purchase). Đây chính là **Class Imbalance**.
> *   Bạn nhận ra: Một người khách vào siêu thị chỉ đi qua tối đa 3-5 gian hàng, họ không thể đi hết cả triệu sản phẩm. Tỷ lệ các sản phẩm bị bỏ quên là cực kỳ lớn. Đây chính là **Matrix Sparsity**.
> *   Bạn thấy siêu thị bán từ cái tăm giá 1.000đ cho tới cái tủ lạnh giá 50 triệu đồng. Sự chênh lệch khổng lồ này khiến việc tính toán trung bình giá bị méo mó. Đây chính là **Skewed Price Distribution**.
> 
> Cuốn sổ tay khảo sát này giúp bạn hiểu rõ "tính nết" của siêu thị để chuẩn bị các công cụ phân loại, nhóm hàng hóa thông minh ở Phase tiếp theo.
