# 10. Tiêu chí Lựa chọn Tập dữ liệu (Dataset Selection Criteria)

Để phục vụ tốt nhất cho bài toán Two-Stage Recommender System (có giả lập tính chất của hệ thống Streaming thông qua Batch processing), tập dữ liệu (Dataset) được chọn để tải về phải đáp ứng các tiêu chuẩn khắt khe sau đây.

## 1. Yêu cầu về Đặc tính Dữ liệu (Data Features)

Tập dữ liệu lý tưởng BẮT BUỘC phải bao gồm 2 file gốc (hoặc 2 bảng) trở lên để thể hiện được tính đa dạng của Feature Store:

### A. Dữ liệu Tương tác Hành vi (Interaction / Event Log)
- **Cột `Timestamp` (Tuyệt đối quan trọng):** Dữ liệu phải có mốc thời gian thực hiện hành động (Epoch time hoặc Datetime) để áp dụng thuật toán **Sessionization** (Gom nhóm hành vi theo phiên 30 phút). Nếu không có timestamp, toàn bộ logic xử lý luồng giả lập sẽ phá sản.
- **Cột `User_ID` & `Item_ID`:** Mã định danh người dùng và sản phẩm cơ bản.
- **Cột `Event_Type`:** Phải chứa đa dạng loại hành vi (Implicit Feedback). Ví dụ: `view` (xem), `cart` (thêm giỏ hàng), `purchase` (mua). 
  - *Tại sao?* Vì chúng ta cần các hành vi `view` không dẫn đến `purchase` để tạo **Pseudo-negative labels** (Nhãn giả 0) cho giai đoạn Ranking. 

### B. Siêu dữ liệu (Metadata / Context)
- **Item Metadata:** Dữ liệu mô tả sản phẩm (Category, Price, Brand, Text Description). Bắt buộc phải có để xây dựng vector nhúng (Embedding) bằng Content-based nhằm giải quyết bài toán Cold-start (Sản phẩm mới chưa ai mua).
- **User Metadata (Tùy chọn):** Tuổi, giới tính, vị trí. (Càng tốt để đưa vào mô hình LightGBM Ranking, nhưng nếu không có thì mô hình sẽ tự học biểu diễn User thông qua lịch sử tương tác của họ).

## 2. Tiêu chí Về Quy mô & Phần cứng (Scale & Hardware)

Dự án được chạy trên máy tính cá nhân sử dụng GPU **NVIDIA RTX 5060** (VRAM khoảng 8GB).
- **Độ thưa thớt (Sparsity):** Ma trận User-Item phải đủ lớn để thể hiện tính thực tế (User chỉ tương tác với < 1% tổng số sản phẩm). 
- **Tổng số dòng tương tác:** Nằm trong khoảng **1 triệu đến 10 triệu dòng (Rows)** (Cỡ file CSV vài trăm MB đến 2GB). Quá nhỏ (< 100k) sẽ không chứng minh được sức mạnh của FAISS Vector Search, quá lớn (> 50GB) sẽ gây Out-Of-Memory (OOM) khi train PyTorch/Spark trên phần cứng Local.

## 3. Các Bộ dữ liệu Đề xuất (Suggested Datasets)

Dựa trên các tiêu chí trên, dưới đây là các bộ dữ liệu công khai trên Kaggle/KDD phù hợp nhất cho Project này:

1. **eCommerce behavior data from multi category store (REES46):**
   - Rất hoàn hảo! Có 7 tháng dữ liệu hành vi.
   - Có `event_type` (`view`, `cart`, `purchase`), có `user_session`, có `product_id`, `category_id`, `brand`, `price`.
   - Đáp ứng 100% tiêu chí tạo Pseudo-label và Sessionization.
2. **Taobao User Behavior Dataset (Alibaba):**
   - Tập dữ liệu tỷ lệ lấy mẫu từ Taobao. Có `User ID`, `Item ID`, `Category ID`, `Behavior type` (pv, buy, cart, fav) và `Timestamp`. Rất tốt để làm thuật toán chuỗi và Two-Tower.
3. **H&M Personalized Fashion Recommendations (Kaggle):**
   - Chứa `transactions_train.csv` (với timestamp), `articles.csv` (metadata sản phẩm cực kỳ phong phú), `customers.csv` (metadata khách hàng). 
   - Điểm yếu: Chỉ có transaction mua hàng, thiếu hành vi `view` (implicit), khó gắn nhãn 0, phải dùng kỹ thuật Negative Sampling ngẫu nhiên thay vì Pseudo-labeling.

**✅ Lời khuyên chốt:** Nên chọn dataset của **REES46 (eCommerce behavior data)** sinh ra để làm đúng bài toán này.