# Phân tích dữ liệu bằng python

### 1. Định hình bài toán và Phương pháp tiếp cận

**1.1. Bản chất bài toán hệ thống gợi ý**
Về mặt toán học, hệ thống gợi ý là bài toán xấp xỉ hàm độ lợi (utility function). Cho tập người dùng U và tập sản phẩm I, mục tiêu là dự đoán hàm f:U×I→R, trong đó R là giá trị thể hiện mức độ quan tâm của người dùng đối với sản phẩm. Hệ thống sẽ đề xuất các sản phẩm i∈I chưa được người dùng u∈U tương tác sao cho f(u,i) đạt giá trị lớn nhất.

Mục tiêu kỹ thuật tối hậu khi triển khai thực tế là tối ưu hóa các hàm mục tiêu kinh doanh thông qua dữ liệu:

- Tăng tỷ lệ nhấp chuột (CTR - Click-Through Rate).
- Tăng tỷ lệ chuyển đổi (CR - Conversion Rate) từ lượt xem sang lượt mua.
- Tối ưu hóa giá trị vòng đời khách hàng (CLV) hoặc thời gian lưu lại nền tảng (Dwell time).

**1.2. Phân tách luồng dữ liệu đầu vào**
Quá trình xây dựng mô hình phụ thuộc hoàn toàn vào 3 luồng dữ liệu cốt lõi:

- **Phản hồi tường minh (Explicit Feedback):** Điểm đánh giá (rating 1-5 sao), bình luận. Dữ liệu này có độ tin cậy cao nhưng độ bao phủ rất thấp (người dùng lười đánh giá).
- **Phản hồi ẩn (Implicit Feedback):** Lịch sử nhấp chuột, thời gian xem sản phẩm, thêm vào giỏ hàng, lịch sử mua. Dữ liệu này dồi dào, liên tục nhưng nhiễu cao. Đây là bài toán One-Class Collaborative Filtering, hệ thống chỉ ghi nhận tín hiệu tích cực (tương tác), sự vắng mặt của tương tác có thể là do người dùng không thích hoặc do họ chưa từng nhìn thấy sản phẩm.
- **Siêu dữ liệu (Metadata):** Đặc trưng người dùng (độ tuổi, giới tính, vị trí) và đặc trưng sản phẩm (nhãn hàng, danh mục, mô tả văn bản, giá cả).

**1.3. Phân loại các kiến trúc gợi ý cốt lõi**
Dựa trên luồng dữ liệu đầu vào, các mô hình được chia thành ba nhánh kiến trúc chính:

- **Content-Based Filtering (Lọc theo nội dung):** Chỉ sử dụng Item metadata và User profile. Mô hình biểu diễn sản phẩm và lịch sử sở thích của người dùng dưới dạng các vector đặc trưng (ví dụ: TF-IDF hoặc Word Embeddings). Mức độ liên quan được tính bằng khoảng cách không gian giữa hai vector (thường dùng Cosine Similarity). Điểm mạnh là không bị ảnh hưởng bởi độ thưa thớt tương tác, điểm yếu là thiếu tính đa dạng (Serendipity), chỉ gợi ý những thứ người dùng đã biết.
- **Collaborative Filtering (Lọc cộng tác):** Chỉ sử dụng ma trận tương tác (User-Item Interaction Matrix). Phân tích hành vi đám đông để tìm ra các người dùng có chung sở thích (User-based) hoặc các sản phẩm thường được tương tác cùng nhau (Item-based). Kỹ thuật cốt lõi ở đây là phân rã ma trận (Matrix Factorization), xấp xỉ ma trận tương tác thô R bằng tích của hai ma trận nhúng mật độ thấp: R≈P×QT (trong đó P là ma trận nhúng người dùng, Q là ma trận nhúng sản phẩm).
- **Hybrid Recommender (Hệ thống lai):** Kết hợp cả hai phương pháp trên để bù trừ khuyết điểm. Thường được thiết kế theo kiế n trúc Two-Tower (một tháp học đặc trưng người dùng, một tháp học đặc trưng sản phẩm) hoặc Ensemble methods (tính trọng số trung bình các đầu ra).

**1.4. Thách thức kỹ thuật đặc thù**
Khi tiền xử lý và huấn luyện, phải giải quyết 3 bài toán kinh điển sau:

- **Cold-start (Khởi động lạnh):** Khi có người dùng mới (chưa có lịch sử) hoặc sản phẩm mới (chưa ai tương tác), ma trận Collaborative Filtering vô tác dụng. Cần xử lý bằng cách fallback về mô hình Content-based hoặc sử dụng các heuristic (gợi ý sản phẩm thịnh hành nhất).
- **Sparsity (Độ thưa thớt dữ liệu):** Trong thực tế, ma trận User-Item có hàng triệu hàng và cột, nhưng 99% giá trị là rỗng (người dùng chỉ tương tác với một phần rất nhỏ sản phẩm). Điều này dễ dẫn đến hiện tượng Overfitting khi huấn luyện mô hình. Cần áp dụng các kỹ thuật giảm chiều dữ liệu (SVD, PCA) hoặc Regularization mạnh.
- **Scalability (Khả năng mở rộng):** Việc tính toán khoảng cách vector từ 1 người dùng đến hàng triệu sản phẩm trong thời gian thực (nhỏ hơn 100ms) là không khả thi. Cần thiết kế pipeline dữ liệu sao cho việc huấn luyện diễn ra offline, và quá trình dự đoán online phải sử dụng các thuật toán tìm kiếm lân cận gần đúng (ANN - Approximate Nearest Neighbor) như FAISS hoặc ScaNN.

[Full Process](https://www.notion.so/Full-Process-33f1e832c6fe8060b71ac683f8471732?pvs=21)