# Component of Data Ingestion process

## 1. Lớp Thu thập và Vận chuyển (Data Ingestion & Transport)

### Data Tracker (SDK/Agent)

- **Dữ liệu đảm nhận:** Các sự kiện thô (Raw Events) phát sinh từ hành động của người dùng trên Client (Click, View, Scroll, Search).
- **Vai trò:** Là điểm tiếp xúc đầu tiên. Nó đóng gói hành vi người dùng kèm theo siêu dữ liệu ngữ cảnh (Timestamp, Device ID, Session ID) và gửi về Server.
- **Nếu không có:** Hệ thống hoàn toàn không có dữ liệu đầu vào về hành vi thực tế của người dùng.

### Message Broker (Apache Kafka)

- **Dữ liệu đảm nhận:** Luồng phản hồi ẩn (Implicit Feedback) có tần suất cực cao và dung lượng lớn.
- **Vai trò:** Đóng vai trò là "vùng đệm" (Buffer) và hệ thống phân phối dữ liệu. Kafka đảm bảo dữ liệu không bị mất nếu các hệ thống phía sau (Flink) bị quá tải hoặc gặp sự cố. Nó cũng cho phép nhiều hệ thống cùng đọc một luồng dữ liệu một lúc.
- **Nếu không có:** Hệ thống dễ bị sập khi lưu lượng truy cập tăng đột biến (Spike). Dữ liệu có thể bị mất mát và không thể mở rộng quy mô xử lý.

### Schema Registry

- **Dữ liệu đảm nhận:** Định nghĩa cấu trúc dữ liệu (Schemas - ví dụ: Avro, Protobuf).
- **Vai trò:** Kiểm soát định dạng dữ liệu truyền qua Kafka. Nó đảm bảo rằng dữ liệu "đầu phát" (Producer) và "đầu nhận" (Consumer) luôn hiểu nhau.
- **Nếu không có:** Rủi ro "Data Corruption" cao. Một thay đổi nhỏ về cấu trúc dữ liệu ở phía App có thể làm hỏng toàn bộ quy trình xử lý phía sau, gây lỗi hệ thống nghiêm trọng.

---

## 2. Lớp Xử lý Luồng (Stream Processing - Cửa ngõ cho Phản hồi ẩn)

### Stream Processor (Apache Flink)

- **Dữ liệu đảm nhận:** Dữ liệu hành vi từ Kafka.
- **Nhiệm vụ cụ thể trong quy trình:**
    - **Sessionization:** Gom nhóm các hành động rời rạc (Click A, Click B) thành một phiên (Session) dựa trên thời gian thực.
    - **Pseudo-labeling (Gắn nhãn giả):** Áp dụng logic để phân định giữa "Người dùng không thích sản phẩm" và "Người dùng chưa nhìn thấy sản phẩm" (Bài toán One-Class CF). Ví dụ: Nếu người dùng lướt qua sản phẩm mà không click, Flink gắn nhãn "Negative" giả để mô hình AI học.
- **Vai trò:** Chuyển đổi dữ liệu thô thành thông tin có ý nghĩa (Features) ngay lập tức.
- **Nếu không có:** Hệ thống không thể phản ứng với ý định hiện tại của người dùng. Gợi ý sẽ bị lỗi thời (chậm vài giờ hoặc vài ngày).

---

## 3. Lớp Xử lý Lô (Batcqqqh Processing - Cửa ngõ cho Metadata & Explicit Feedback)

### Data Warehouse (DW) / Data Lake

- **Dữ liệu đảm nhận:** Phản hồi tường minh (Rating, Review), Siêu dữ liệu (Thông tin sản phẩm, Profile người dùng).
- **Vai trò:** Nơi lưu trữ tập trung dữ liệu "tĩnh" và dữ liệu lịch sử lâu dài. Đây là nguồn dữ liệu có độ tin cậy cao nhất.
- **Nếu không có:** Hệ thống thiếu đi cái nhìn sâu sắc về bản chất đối tượng (ví dụ: không biết sản phẩm thuộc phân khúc nào, người dùng có sở thích dài hạn ra sao).

### Batch Processor (Apache Spark)

- **Dữ liệu đảm nhận:** Dữ liệu khổng lồ từ Data Warehouse.
- **Vai trò:** Thực hiện các phép Join phức tạp và tính toán đặc trưng (Feature Engineering) quy mô lớn. Spark tạo ra các tập dữ liệu huấn luyện (Training Datasets) chính xác bằng cách kết hợp hành vi quá khứ với các thuộc tính sản phẩm.
- **Nếu không có:** Không thể huấn luyện được các mô hình AI mạnh mẽ. Hệ thống chỉ có thể thực hiện các gợi ý đơn giản dựa trên quy tắc (Rule-based).
- Quy trình tính toán với Apache Spark
- **Distributed Computing:** Spark sử dụng cơ chế **In-memory computing** và chia nhỏ công việc thành các **Partition** để chạy song song trên cụm server (Cluster). Khác với Flink xử lý từng sự kiện, Spark xử lý một khối lượng dữ liệu khổng lồ (Petabytes) trong một chu kỳ (Batch).
- **ETL (Extract, Transform, Load):** Trích xuất dữ liệu từ các nguồn (App log, Transaction DB), biến đổi chúng (chuẩn hóa định dạng, xử lý dữ liệu thiếu) và nạp vào kho lưu trữ tập trung.
- **ELT (Extract, Load, Transform):** Xu hướng hiện đại hơn, nạp dữ liệu thô vào các kho dữ liệu đám mây mạnh mẽ (như BigQuery, Snowflake) rồi mới dùng SQL để biến đổi dữ liệu bên trong đó.

---

## 4. Lớp Tích hợp và Phục vụ (Integration & Serving)

### Feature Store

- **Dữ liệu đảm nhận:** Các Vector đặc trưng (Features) đã được xử lý xong từ cả Flink (Online) và Spark (Offline).
- **Vai trò:** Là "kho chứa chung" duy nhất. Nó đảm bảo mô hình AI khi huấn luyện (dùng dữ liệu Spark) và khi dự đoán thực tế (dùng dữ liệu Flink) luôn nhận được định dạng và logic dữ liệu giống hệt nhau.
- **Nếu không có:** Xảy ra hiện tượng "Training-Serving Skew". Mô hình học một kiểu nhưng thực tế lại gặp dữ liệu kiểu khác, dẫn đến kết quả gợi ý sai lệch hoàn toàn dù mô hình có vẻ rất thông minh.

### Labeling Engine (Thành phần bổ trợ cho Pseudo-labels)

- **Dữ liệu đảm nhận:** Logic nghiệp vụ và dữ liệu phản hồi ẩn.
- **Vai trò:** Cụ thể hóa các quy tắc gắn nhãn giả cho mô hình Collaborative Filtering. Nó xác định các ngưỡng (thresholds) để coi một sự vắng mặt tương tác là tín hiệu tiêu cực.
- **Nếu không có:** Mô hình AI sẽ bị thiên kiến (bias) cực nặng vào những thứ người dùng đã click, không thể mở rộng phạm vi gợi ý sang các sản phẩm mới.

---

### Tổng kết luồng dữ liệu

1. **Hành vi ẩn** -> Tracker -> Kafka -> **Flink** (Gom nhóm/Gắn nhãn giả) -> Feature Store.
2. **Hành vi tường minh/Siêu dữ liệu** -> Data Warehouse -> **Spark** (ETL/Feature Engineering) -> Feature Store.
3. **Mô hình AI** truy cập Feature Store để lấy dữ liệu đã được tiền xử lý hoàn chỉnh nhằm thực hiện huấn luyện hoặc đưa ra gợi ý trực tiếp.