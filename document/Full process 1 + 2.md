# Full process 1 + 2

---

## 1. Giai đoạn 1: Tiếp nhận và Phân luồng dữ liệu (Data Ingestion)

Hệ thống bắt đầu bằng việc thu nhận dữ liệu từ các nguồn khác nhau, được phân tách dựa trên tính chất của dữ liệu:

- **Luồng nóng (Hot Path - Phản hồi ẩn):** Các sự kiện Clickstream (click, view) được **Tracker** thu thập và đẩy ngay vào **Apache Kafka**. Do đặc thù dữ liệu này có tần suất cực lớn (hàng triệu sự kiện mỗi giây), Kafka đóng vai trò vùng đệm để đảm bảo không mất mát dữ liệu và cho phép các hệ thống phía sau tiêu thụ theo khả năng của chúng.
- **Luồng nguội (Cold Path - Phản hồi tường minh & Siêu dữ liệu):** Dữ liệu như đánh giá (rating), bình luận (review) và thông tin sản phẩm (metadata) thường có độ tin cậy cao và ít thay đổi hơn. Chúng được lưu trữ trong các cơ sở dữ liệu quan hệ (RDBMS) và được trích xuất (Extract) định kỳ vào **Data Warehouse** hoặc **Data Lake**.

---

## 2. Giai đoạn 2: Tiền xử lý tại lớp Stream và Batch

Mục tiêu của giai đoạn này là làm sạch và cấu trúc lại dữ liệu thô thành thông tin có ngữ cảnh.

### Tại lớp Stream (Apache Flink)

- **Sessionization (Gom nhóm theo phiên):** Flink sử dụng **Session Window** để nhóm các hành động rời rạc của một người dùng. Nếu khoảng cách giữa hai hành động vượt quá một ngưỡng (ví dụ: 30 phút), một phiên mới sẽ được tạo. Điều này giúp xác định ý định mua sắm tức thời của người dùng.
- **Xử lý sự kiện muộn (Watermark):** Flink sử dụng cơ chế Watermark để chờ các dữ liệu bị lag do mạng. Điều này đảm bảo tính toàn vẹn của phiên: các click xảy ra lúc 10:00 nhưng đến server lúc 10:05 vẫn được gộp đúng vào phiên 10:00.

### Tại lớp Batch (Apache Spark)

- **Data Enrichment (Làm giàu dữ liệu):** Spark thực hiện các phép Join quy mô lớn giữa dữ liệu hành vi lịch sử và bảng siêu dữ liệu. Ví dụ: Từ một ID sản phẩm, Spark sẽ gắn thêm các thông tin như thương hiệu, danh mục, phân khúc giá.
- **Deduplication (Khử trùng lặp):** Loại bỏ các bản ghi lỗi hoặc trùng lặp trong quá trình lưu trữ lâu dài tại Data Warehouse.

---

## 3. Giai đoạn 3: Logic gắn nhãn giả (Pseudo-labeling)

Đây là bước chuẩn bị dữ liệu cho bài toán **One-Class Collaborative Filtering (OCCF)**, vì trong thực tế chúng ta thường chỉ thấy những gì người dùng thích, chứ không thấy rõ những gì họ ghét.

- **Tín hiệu tích cực (Positive Signals):** Các hành động như Click hoặc Add-to-cart được mặc định gắn nhãn $1$.
- **Tín hiệu tiêu cực giả (Pseudo-negative Labels):** Hệ thống áp dụng chiến lược **Negative Sampling**. Nếu một sản phẩm được hiển thị (Impression) nhưng người dùng không tương tác trong suốt phiên đó, hệ thống sẽ gắn nhãn giả là $0$.
- **Mục đích:** Tạo ra một tập dữ liệu cân bằng để mô hình AI có thể phân biệt được ranh giới giữa sở thích và sự thờ ơ của người dùng.

---

## 4. Giai đoạn 4: Vector hóa và Nhúng đặc trưng (Vectorization)

Đây là giai đoạn quan trọng nhất để chuyển đổi dữ liệu từ dạng thô sang dạng toán học mà máy tính có thể xử lý.

- **Vector hóa siêu dữ liệu (Categorical Vectorization):** Các thông tin như giới tính, khu vực địa lý hoặc danh mục sản phẩm được chuyển đổi bằng kỹ thuật **One-hot Encoding** hoặc **Hashing**.
- **Embedding (Nhúng đặc trưng):** * Các thông tin phi cấu trúc như mô tả sản phẩm (văn bản) hoặc hình ảnh được đưa qua các mô hình tiền huấn luyện (như BERT cho văn bản hoặc ResNet cho hình ảnh) để tạo ra các **Vector Embedding** (ví dụ: một mảng 128 chiều các số thực).
    - Các vector này biểu diễn sản phẩm trong một không gian đa chiều (Latent Space). Những sản phẩm có tính chất tương đồng sẽ có các vector nằm gần nhau về mặt toán học (tính bằng Cosine Similarity).
- **Tính nhất quán:** Bước này được thực hiện đồng bộ trên cả Spark (cho dữ liệu huấn luyện) và Flink (cho dữ liệu thực thi) để đảm bảo các vector luôn có cùng định dạng và ý nghĩa.

---

## 5. Giai đoạn 5: Lưu trữ đặc trưng và Phục vụ (Feature Store)

Cuối cùng, các vector đặc trưng và nhãn được đưa vào **Feature Store** để sử dụng.

- **Materialization (Vật chất hóa):**
    - **Online Store (Redis/Cassandra):** Lưu trữ các vector đặc trưng mới nhất từ Flink để phục vụ việc gợi ý sản phẩm ngay khi người dùng vừa có hành động mới (độ trễ thấp).
    - **Offline Store (S3/Data Warehouse):** Lưu trữ toàn bộ lịch sử các vector đặc trưng và nhãn giả từ Spark để phục vụ việc huấn luyện lại mô hình AI định kỳ.
- **Point-in-time Join:** Khi cần huấn luyện, Feature Store sẽ khớp nối chính xác vector đặc trưng của người dùng tại đúng thời điểm họ thực hiện hành vi trong quá khứ, tránh rò rỉ dữ liệu từ tương lai vào quá trình học.

**Kết quả cuối cùng:** Hệ thống cung cấp một luồng dữ liệu sạch, đã được vector hóa và gắn nhãn, sẵn sàng để các thuật toán Machine Learning thực hiện việc dự đoán sản phẩm phù hợp nhất cho người dùng.