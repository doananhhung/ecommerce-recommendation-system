# Full Process

### **1. Thu thập và Tiền xử lý luồng dữ liệu (Data Ingestion & Preprocessing**

Hệ thống tiếp nhận đồng thời 3 luồng dữ liệu: phản hồi tường minh, phản hồi ẩn và siêu dữ liệu.

- **Xử lý luồng (Stream Processing):** Dữ liệu phản hồi ẩn (click, view) có khối lượng lớn, nhiễu cao, được đẩy qua message broker như Apache Kafka.
- **Xử lý lô (Batch Processing):** Phản hồi tường minh và siêu dữ liệu (độ tin cậy cao, tĩnh hơn) được trích xuất định kỳ từ Data Warehouse.
- **Tiền xử lý:** Gom nhóm hành vi theo phiên (session-based). Với phản hồi ẩn, hệ thống gắn nhãn giả (pseudo-labels) để xử lý bài toán One-Class Collaborative Filtering, phân định tín hiệu tích cực và sự vắng mặt tương tác.

### **2. Trích xuất đặc trưng và Feature Store (Feature Engineering)**

Chuyển đổi dữ liệu thô thành không gian vector.

- **Vector hóa:** Siêu dữ liệu người dùng và sản phẩm được chuyển thành các vector đặc trưng bằng các kỹ thuật nhúng (Word Embeddings, TF-IDF).
- **Feature Store:** Các đặc trưng này được lưu trữ tập trung tại Feature Store (như Feast hoặc Hopsworks) để đảm bảo sự đồng nhất tuyệt đối về mặt dữ liệu giữa môi trường huấn luyện (offline) và môi trường phục vụ (online).

[Component of Data Ingestion process ](Component%20of%20Data%20Ingestion%20process%203421e832c6fe806ebc74ec78bbea186b.md)

[Full process 1 + 2](Full%20process%201%20+%202%203431e832c6fe804cb18beb7db9590be6.md)

[Real process 1 + 2 (Tức là bỏ qua việc thu thập dữ liệu từ web)](Real%20process%201%20+%202%20(T%E1%BB%A9c%20l%C3%A0%20b%E1%BB%8F%20qua%20vi%E1%BB%87c%20thu%20th%E1%BA%ADp%20d%E1%BB%AF%203431e832c6fe800d9455e7a22ba85590.md)

[Explain about relationship between Flink/kafka, Data warehouse và Feature Store](Explain%20about%20relationship%20between%20Flink%20kafka,%20Da%203431e832c6fe8001a192fdb2851788e4.md)

### **3. Huấn luyện ngoại tuyến hai giai đoạn (Offline Training)**

Đường ống huấn luyện định kỳ (thường là hàng ngày) để cập nhật trọng số mô hình.

- **Giai đoạn Recall (Truy xuất):** Sử dụng phân rã ma trận để xấp xỉ ma trận tương tác thô $R$ thành $R\approx P\times Q^T$ hoặc sử dụng kiến trúc Two-Tower. Áp dụng Regularization mạnh hoặc các kỹ thuật giảm chiều dữ liệu (SVD, PCA) để triệt tiêu hiện tượng Overfitting do ma trận thưa thớt (Sparsity). Kết quả đầu ra là ma trận nhúng $P$ (người dùng) và $Q$ (sản phẩm).
- **Giai đoạn Rank (Xếp hạng):** Huấn luyện một mô hình học máy dạng cây (LightGBM/XGBoost) dựa trên nhãn thực tế (đã mua/không mua) để tối ưu hóa các hàm mục tiêu như CTR hay CR.

### **4. Lập chỉ mục không gian Vector (Vector Indexing)**Giải quyết nút thắt về khả năng mở rộng (Scalability).

- Ma trận $Q$ (hàng triệu vector sản phẩm) không thể duyệt tuần tự. Chúng được đẩy vào các công cụ tìm kiếm lân cận gần đúng (ANN) như FAISS hoặc ScaNN.
- Chỉ mục này được tải lên bộ nhớ RAM của các cụm máy chủ phục vụ (Serving Clusters) để đảm bảo tốc độ truy xuất.

### **5. Đường ống phục vụ dự đoán trực tuyến (Online Serving Pipeline)**

Quá trình diễn ra trong thời gian thực (<100ms)  khi người dùng mở ứng dụng.

- **Bước 5.1:** Lấy ID người dùng, truy xuất vector $P_u$ tương ứng từ bộ đệm (Redis). Nếu là người dùng mới (Cold-start), hệ thống chuyển hướng (fallback) sang dùng heuristic (sản phẩm thịnh hành) hoặc Content-based.
- **Bước 5.2 (Recall):** Truy vấn vector $P_u$ vào hệ thống FAISS/ScaNN để lấy nhanh 200 vector sản phẩm gần nhất trong không gian.
- **Bước 5.3 (Rank):** Truy xuất đặc trưng ngữ cảnh hiện tại từ Feature Store. Đẩy 200 ứng viên qua mô hình Rank (LightGBM) để chấm điểm và sắp xếp lại thứ tự từ cao xuống thấp.
- **Bước 5.4:** Áp dụng các bộ lọc nghiệp vụ (loại bỏ sản phẩm hết hàng, đa dạng hóa danh mục) và trả danh sách 20 sản phẩm cuối cùng qua API.

**6. Cập nhật thời gian thực và Giám sát (Real-time Streaming & Monitoring)**

- Hệ thống liên tục tiêu thụ dữ liệu từ Kafka để cập nhật "nhẹ" (incremental update) các vector nhúng của người dùng ngay trong phiên hoạt động.
- Giám sát độ lệch dữ liệu (Data Drift) và suy giảm hiệu suất (Model Decay) để kích hoạt quá trình huấn luyện lại toàn phần ở Bước 3.