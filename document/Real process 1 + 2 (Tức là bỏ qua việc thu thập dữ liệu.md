# Real process 1 + 2 (Tức là bỏ qua việc thu thập dữ liệu từ web)

## How process be like when use existing dataset ?

1. **Phân tích tính cần thiết của Kafka và Flink trong kịch bản Dataset tĩnh.**
2. **Quy trình xử lý dữ liệu Offline (Batch-only Workflow).**
3. **Vị trí và nhiệm vụ của Feature Store khi không có luồng Stream.**
4. **Cách thực hiện Vector hóa và Gắn nhãn trong môi trường Batch.**
5. **Lưu ý về tính hoàn thiện của kiến trúc trong báo cáo khoa học.**

## Detail

### 1. Phân tích tính cần thiết của Kafka và Flink trong kịch bản Dataset tĩnh

Khi bạn sử dụng một Dataset có sẵn (ví dụ: file CSV, Parquet hoặc SQL dump) thay vì thu thập dữ liệu từ trang web thực tế, bản chất dữ liệu của bạn đã là **dữ liệu tĩnh (Bounded Data)**.

- **Kafka:** Nhiệm vụ chính là hệ thống xếp hàng và vận chuyển dữ liệu thời gian thực. Với Dataset tĩnh, dữ liệu đã nằm yên trong ổ đĩa, không cần một "đường ống" trung chuyển như Kafka.
- **Flink:** Nhiệm vụ chính là xử lý các cửa sổ thời gian (Windows) và trạng thái (State) ngay khi dữ liệu đổ về. Khi dữ liệu đã có sẵn toàn bộ lịch sử, bạn có thể sử dụng các công cụ xử lý lô (Batch Processing) để tính toán mọi thứ một lần.

**Kết luận:** Bạn hoàn toàn có thể bỏ qua Kafka và Flink để giảm độ phức tạp của hạ tầng trong giai đoạn này.

### 2. Quy trình xử lý dữ liệu Offline (Batch-only Workflow)

Quy trình của bạn sẽ chuyển đổi sang mô hình xử lý tập trung vào tốc độ huấn luyện thay vì tốc độ phản ứng:

1. **Nguồn dữ liệu:** Dataset (CSV/Parquet) lưu trữ trên Local Disk hoặc Cloud Storage (S3/GCS).
2. **Công cụ xử lý:** Sử dụng **Apache Spark** hoặc **Python (Pandas/Dask)** để đọc toàn bộ tập dữ liệu.
3. **Tiền xử lý & Gom nhóm:** Thực hiện tính toán các đặc trưng (Features) dựa trên toàn bộ lịch sử (ví dụ: tính tổng click của User A trong toàn bộ dataset).
4. **Lưu trữ:** Đưa kết quả vào Feature Store (chủ yếu là Offline Store).

### 3. Vị trí và nhiệm vụ của Feature Store khi không có luồng Stream

Trong kịch bản này, Feature Store không còn đóng vai trò "đồng bộ hóa" giữa hai luồng nữa, nhưng nó vẫn giữ các vai trò cốt lõi sau:

- **Quản lý danh mục đặc trưng (Feature Registry):** Định nghĩa rõ ràng đặc trưng `user_embedding` hay `product_vector` được tạo ra như thế nào để các thành viên khác hoặc chính bạn sau này có thể tái sử dụng.
- **Lưu trữ đặc trưng đã tính toán (Offline Store):** Lưu trữ các vector và nhãn đã xử lý xong dưới dạng bảng để mô hình AI truy xuất trực tiếp khi huấn luyện, thay vì phải chạy lại code tiền xử lý từ đầu mỗi khi train.
- **Tạo tập Training (Dataset Generation):** Giúp bạn thực hiện phép Join giữa các đặc trưng và nhãn (labels) một cách nhất quán theo thời gian.

### 4. Cách thực hiện Vector hóa và Gắn nhãn trong môi trường Batch

Vì không có Flink, các bước này sẽ được thực hiện như sau:

- **Gắn nhãn giả (Pseudo-labeling):** Bạn thực hiện trực tiếp trên bảng dữ liệu bằng cách quét các bản ghi "Impression" (lần hiển thị) mà không có "Click" tương ứng trong cùng một phiên của Dataset để gán nhãn 0.
- **Vector hóa & Embedding:** Bạn sử dụng các thư viện như Scikit-learn, PyTorch hoặc TensorFlow để chuyển đổi các cột siêu dữ liệu (Metadata) thành Vector. Quá trình này diễn ra ngay trong tiến trình xử lý Batch (Spark/Python Job) trước khi đẩy dữ liệu vào Feature Store.

### 5. Lưu ý về tính hoàn thiện của kiến trúc trong báo cáo khoa học

Vì bạn đang thực hiện đồ án tại Đại học Phenikaa, có một lưu ý nhỏ về mặt học thuật:

- **Về mặt thực hành:** Bỏ Kafka/Flink giúp bạn tập trung vào thuật toán và độ chính xác của mô hình AI.
- **Về mặt kiến trúc (Architecture):** Trong báo cáo, bạn vẫn nên mô tả đầy đủ quy trình có Kafka/Flink như một "Kiến trúc mục tiêu" (Target Architecture) dành cho hệ thống thực tế. Bạn có thể giải thích rằng: *"Trong phạm vi nghiên cứu này, do sử dụng Dataset tĩnh, các thành phần Stream Processing được giả lập bằng quy trình xử lý Batch để tối ưu hóa tài nguyên tính toán."*