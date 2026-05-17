# 6. Luồng hệ thống tổng thể (System Processing Flow)

## 1. Luồng Offline (Phục vụ Huấn luyện - Training)
Do Project chạy với static dataset (bỏ qua Kafka/Flink), quy trình offline định kỳ sẽ chạy như sau:
1. Load dataset từ ổ cứng qua `Pandas` / `Spark`.
2. Biến đổi dữ liệu, gom nhóm phiên, gán nhãn giả (thành file Training Dataset).
3. Đẩy vào PyTorch (2.7+) huấn luyện **Mô hình Recall** -> Lưu các vector nhúng (Embedding) Item ra file và nạp cục bộ vào bộ nhớ FAISS.
4. Lấy output sinh ra từ Recall (các feature kết hợp), bổ sung ngữ cảnh, train tiếp **Mô hình Rank (LightGBM)**. Trích xuất mô hình cây ra file.

## 2. Luồng Online (Phục vụ Suy diễn - Inference / Serving)
Khi có Request "Hãy gợi ý cho tôi!" từ 1 User (chạy < 100ms):
1. Nhận `User_ID`.
2. **Cold-start Check:** Nếu User không tồn tại, trả về Trending list.
3. Nếu hợp lệ, lấy vector tĩnh $P_u$ của người dùng.
4. **Recall Lookup:** Truy vấn FAISS Index -> Trả về danh sách `[Item_ID_1, Item_ID_2... Item_ID_200]`.
5. **Feature Fetch:** Lấy toàn bộ vector đặc trưng của User và 200 Items từ Feature Store (đã lưu sẵn ở ổ cứng hoặc Redis cache) nối vào nhau.
6. **Ranking:** Cho 200 dữ liệu bảng này chạy qua Lightweight LightGBM (đã train trên GPU RTX5060) lấy ra danh sách xác suất rác. Lấy Top 20 xếp cao nhất.
7. **Business Rules:** Áp dụng bộ lọc bộ kinh doanh (ví dụ: đã hết hàng, ẩn các sp trùng lặp thư mục).
8. Return Array kết quả và kết thúc phiên.