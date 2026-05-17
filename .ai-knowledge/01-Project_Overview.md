# 1. Tổng quan Project (Project Overview)

## Định nghĩa & Mục tiêu
- **Tên dự án:** Hệ thống gợi ý sản phẩm (Recommender System) 2 giai đoạn (Two-Stage).
- **Mục tiêu kỹ thuật:** Xây dựng hệ thống dự đoán sở thích người dùng, đề xuất các sản phẩm tiềm năng để tối ưu hóa tỷ lệ nhấp chuột (CTR) và tỷ lệ chuyển đổi (CR).
- **Mục tiêu học thuật:** Xây dựng project phục vụ đồ án Đại học (Phenikaa), báo cáo được thiết kế theo chuẩn Realtime/Streaming (Kafka/Flink), nhưng quá trình thực hành thực tế sử dụng phương pháp Batch Processing trên Static Dataset để giảm tải hạ tầng.

## Triết lý thiết kế (Philosophy)
- **Scale-first:** Phải xử lý được ở mức độ hàng triệu người dùng/sản phẩm, giải quyết bài toán tính toán siêu lớn (Scalability).
- **Two-Stage Approach:** Không dùng một mô hình khổng lồ duy nhất. Chia nhỏ thành 2 giai đoạn:
  - *Recall (Truy xuất):* Lọc nhanh từ hàng triệu sản phẩm xuống còn ~200 sản phẩm tiềm năng nhất.
  - *Rank (Xếp hạng):* Chấm điểm kỹ lưỡng và sắp xếp 200 sản phẩm này để chọn ra 20 sản phẩm tốt nhất.

## Hướng thực hiện cốt lõi
- Xử lý dữ liệu đa luồng (Phản hồi ẩn, Phản hồi tường minh, Siêu dữ liệu).
- Giải quyết bài toán thưa thớt (Sparsity) và Cold-start trong Recommender System.
- Thay vì làm Realtime Data Pipeline phức tạp, hệ thống rút gọn bằng cách làm Batch offline trên tập dữ liệu có sẵn để dồn toàn lực vào tinh chỉnh mô hình và thuật toán AI.