# Phase 2: Đường Ống Dữ Liệu (Data Pipeline) & Thiết Kế Đặc Trưng Point-in-time

Trong các hệ thống học máy thực tế, dữ liệu thô không bao giờ có sẵn ở dạng bảng đặc trưng hoàn hảo để đưa thẳng vào mô hình. Phase 2 là giai đoạn xây dựng hệ thống tự động hóa xử lý dữ liệu (Data Pipeline), làm sạch thông tin, gán trọng số hành vi và thực hiện kỹ thuật quan trọng nhất của hệ thống gợi ý dạng chuỗi thời gian: **Point-in-time Feature Engineering** nhằm loại bỏ hoàn toàn lỗi rò rỉ dữ liệu (Data Leakage) và bắt trọn hành vi phiên (session dynamics).

---

## 🎯 Mục Tiêu Của Phase 2

1.  **Phân chia phiên hoạt động (Sessionization)**: Nhóm các hành động rời rạc của một người dùng thành các phiên mua sắm logic dựa trên thời gian ngắt quãng 30 phút.
2.  **Định lượng hành vi ngầm (Implicit Feedback)**: Gán trọng số giá trị thích hợp cho từng loại hành vi của khách hàng thay vì chỉ coi chúng là nhị phân (0 hoặc 1).
3.  **Thiết kế đặc trưng Point-in-time**: Tính toán các chỉ số thống kê của User và Item tại đúng thời điểm tương tác xảy ra, đảm bảo mô hình không "nhìn trước tương lai" (chống Time Travel).
4.  **Tích hợp đặc trưng cấp phiên (Session-level Features)**: Đo lường mức độ tích cực của người dùng trong phiên mua sắm và độ phổ biến của sản phẩm theo phiên.
5.  **Chuẩn bị Feature Store Snapshot**: Tạo sẵn các bảng tra cứu đặc trưng mới nhất của người dùng và sản phẩm để phục vụ tức thì cho hệ thống suy luận API trực tuyến.

---

## 📁 Thư Mục Khởi Tạo & Vai Trò

Trong Phase này, các đoạn mã nguồn xử lý dữ liệu được thiết kế theo dạng module hóa trong thư mục `src/data_pipeline/`:

```
EDA_project/
├── data/
│   ├── sessions/
│   │   └── labeled_sessions.parquet   # Dữ liệu hành vi đã được phân chia phiên và gán nhãn
│   └── feature_store/
│       ├── user_features.parquet      # Đặc trưng tĩnh mới nhất của User phục vụ Online Serving
│       └── item_features.parquet      # Đặc trưng tĩnh mới nhất của Item phục vụ Online Serving
└── src/
    └── data_pipeline/
        ├── __init__.py
        ├── sessionizer.py             # Thuật toán chia phiên hành vi theo thời gian
        ├── pseudo_label.py            # Hàm chuyển đổi hành vi thành trọng số tương tác
        ├── featurizer.py              # Công cụ tính toán đặc trưng Point-in-time lũy kế
        └── run_pipeline.py            # Nhạc trưởng điều phối toàn bộ đường ống dữ liệu
```

### 🔹 Mục đích chi tiết của từng tệp:

*   **`sessionizer.py`**:
    *   *Kỹ thuật*: Khách hàng có thể vào trang thương mại điện tử nhiều lần trong ngày. Tệp này sử dụng thuật toán sắp xếp lịch sử tương tác của từng User theo thời gian, tính khoảng cách giữa hai hành động liên tiếp. Nếu khoảng cách này **vượt quá 30 phút** (mặc định), hệ thống sẽ tự động ngắt và đánh số một mã phiên mới (`custom_session_id`).
    *   *Ý nghĩa bối cảnh*: Bắt kịp "Ý định mua sắm ngắn hạn" (Short-term Intent) và Khử nhiễu sở thích dài hạn. Nếu không chia phiên, hệ thống sẽ trộn đều các hành vi tìm son môi ngắn hạn vào sở thích công nghệ dài hạn của người dùng, làm loãng trải nghiệm xem đồ công nghệ của họ vào ngày hôm sau.
*   **`pseudo_label.py`**:
    *   *Kỹ thuật*: Lượt xem (`view`) thể hiện sự tò mò nhẹ, bỏ vào giỏ (`cart`) thể hiện sự quan tâm mạnh mẽ, và mua hàng (`purchase`) thể hiện sự cam kết tối đa. Thay vì gán nhãn đơn giản 0 hoặc 1, file này chuyển đổi cột `event_type` thành một điểm số liên tục (Implicit Weight): **View = 0.1, Cart = 0.5, Purchase = 1.0**. Nhãn này sẽ được dùng trực tiếp để mô hình học mức độ ưu tiên.
*   **`featurizer.py`**:
    *   *Kỹ thuật*: Chứa hai hàm vô cùng quan trọng:
        1.  `add_point_in_time_features`: Tính toán đặc trưng lịch sử của User/Item tại thời điểm tương tác diễn ra (Point-in-Time). Đặc biệt, hàm này tính toán các đặc trưng phiên lũy tiến mà không bị rò rỉ dữ liệu (xem chi tiết ở mục dưới).
        2.  `extract_user_features` / `extract_item_features`: Trích xuất trạng thái đặc trưng cuối cùng (Latest Snapshot) của từng User và Item tại cuối dòng thời gian, lưu xuống thư mục `feature_store` để API nạp lên RAM tra cứu nhanh.
*   **`run_pipeline.py`**:
    *   *Kỹ thuật*: Script đầu dẫn (Orchestrator). Đọc dữ liệu thô, chạy tuần tự qua bộ phân phiên `sessionizer`, bộ gán nhãn `pseudo_label`, bộ trích xuất đặc trưng `featurizer`, chia tập Train/Test theo dòng thời gian (Time-based split) và lưu lại các tệp Parquet.

---

## 🛠️ Thiết Kế Đặc Trưng Point-in-time Cấp Phiên (Session Features Design)

Để tăng độ nhạy bén của mô hình với bối cảnh phiên hoạt động thời gian thực, chúng ta bổ sung các đặc trưng cấp phiên được tính toán lũy kế point-in-time như sau:

### 1. `user_session_interaction_count` (Số tương tác trong phiên hiện tại)
*   *Mô tả*: Đếm số lượng hành động (view, cart, purchase) của người dùng đó trong phiên cụ thể này tính đến trước thời điểm tương tác $t$.
*   *Công thức Point-in-time*:
    ```python
    result["user_session_interaction_count"] = result.groupby(["user_id", "custom_session_id"]).cumcount()
    ```
*   *Mục đích*: Nếu người dùng đã click xem liên tiếp 15 sản phẩm trong 10 phút qua ở phiên hiện tại, chỉ số này tăng cao chứng tỏ họ đang có nhu cầu mua sắm cực kỳ tích cực. Mô hình sẽ tăng độ nhạy để thúc đẩy các gợi ý chuyển đổi (mua hàng).

### 2. `item_session_popularity` (Độ phổ biến sản phẩm theo phiên)
*   *Mô tả*: Tổng số lượng phiên độc nhất (unique sessions) trong lịch sử của toàn bộ hệ thống đã tương tác với sản phẩm này lũy kế tính đến trước thời điểm tương tác $t$.
*   *Công thức Point-in-time*:
    ```python
    first_item_session = ~result.duplicated(["product_id", "custom_session_id"])
    result["item_session_popularity"] = (
        first_item_session.groupby(result["product_id"]).cumsum()
        - first_item_session.astype(int)
    )
    ```
*   *Mục đích*: Đo lường xem sản phẩm này có đang là "xu hướng" xuất hiện trong nhiều giỏ hàng/phiên xem của mọi người hay không. Việc đếm theo số phiên độc nhất thay vì lượt click thô giúp triệt tiêu độ nhiễu từ các hành vi spam click (click nhiều lần một sản phẩm trong cùng một phiên).

---

## 🛠️ Công Nghệ & Khái Niệm Kỹ Thuật Sử Dụng

1.  **Tệp định dạng Parquet**: Sử dụng thư viện `pyarrow` để ghi dữ liệu. Parquet lưu trữ dữ liệu theo cột (columnar format) và nén cực tốt, giúp tốc độ đọc ghi nhanh gấp 5-10 lần và tiết kiệm 70% dung lượng đĩa so với file `.csv` truyền thống.
2.  **Point-in-Time (Chống Data Leakage)**:
    *   *Kỹ thuật*: Tính toán tất cả các biến số lũy kế (interactions, sessions, prices) tính đến đúng thời điểm $t$ trước khi sự kiện diễn ra. Tuyệt đối không dùng các phép toán gộp nhóm tương lai (như `groupby.size()`), đảm bảo mô hình không học "nhìn trước tương lai", mang lại độ ổn định tối đa khi mang mô hình ra Serving thực tế.
3.  **Log Transformation cho Price**: 
    *   Sử dụng công thức $y = \log(x + 1)$ để nén khoảng cách giữa sản phẩm giá rẻ và sản phẩm siêu đắt đỏ, giúp các mô hình Gradient Boosting (LightGBM) hội tụ nhanh hơn và không bị lệch bởi các giá trị ngoại lai (outliers).

---

## 💡 Hình Dung Trực Quan (Ví Dụ Dễ Hiểu)

> **Ví dụ về "Cuốn sổ nhật ký của Thám tử Tư cấp Phiên"**
> 
> Hãy tưởng tượng thám tử tư theo dõi một khách hàng trong siêu thị.
> 
> Nếu thám tử viết báo cáo cuối ngày là: *"Khách hàng A hôm nay đã mua 10 món đồ, do đó tại thời điểm 9 giờ sáng lúc họ xem gói kẹo, tôi biết chắc họ là khách hàng vip và sẽ mua gói kẹo đó"*. Báo cáo này hoàn toàn vô dụng ngoài đời thực. Đây chính là **Rò rỉ dữ liệu tương lai**.
> 
> **Phase 2** giải quyết việc này bằng cách bắt thám tử ghi nhật ký chi tiết từng giây từng phút (`featurizer.py`):
> *   *Lúc 9:00*: Khách vào siêu thị (Phiên thứ nhất bắt đầu - `sessionizer.py`). Khách xem gói kẹo. Nhật ký ghi: *"Số tương tác lũy kế trước đó của User trong phiên này là 0. Gói kẹo này có số lượng phiên chứa nó trước đó là 150 (item_session_popularity)"*.
> *   *Lúc 9:15*: Khách bỏ gói kẹo vào giỏ hàng. Nhật ký ghi nhận nhãn tương tác mạnh hơn: **0.5 điểm** (`pseudo_label.py`).
> *   *Lúc 9:20*: Khách tiếp tục xem thanh sô-cô-la thứ hai trong phiên. Nhật ký ghi: *"Số tương tác lũy kế trước đó của User trong phiên này tăng lên 2 (gồm xem kẹo và bỏ kẹo vào giỏ) - `user_session_interaction_count = 2`"*.
> 
> Nhờ cuốn nhật ký ghi chép nghiêm ngặt theo thời gian thực này, mô hình AI ở các Phase sau sẽ được học một cách trung thực nhất, giống hệt với cách nó phải đưa ra dự đoán ngoài đời thực khi API chạy trực tuyến.
