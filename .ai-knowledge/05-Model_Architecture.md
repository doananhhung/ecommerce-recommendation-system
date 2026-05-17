# 5. Kiến trúc Mô hình Học máy (Model Architecture)

Hệ thống được thiết kế theo cấu trúc **Two-Stage Recommender Pipeline** phổ biến trong quy mô công nghiệp (Shopee, Netflix, Tiktok...).

## Giai đoạn 1: RECALL (Truy xuất ứng viên)
- **Mục tiêu:** Cắt giảm không gian tìm kiếm từ hàng triệu item xuống còn khoảng 200 items có mức độ liên quan cơ bản (High coverage, low precision).
- **Mô hình tiếp cận:**
  - **Collaborative Filtering (Matrix Factorization) hoặc Two-Tower:** Xấp xỉ ma trận User-Item Interaction.
  - Phân rã ma trận $R \approx P \times Q^T$.
  - Biến User và Item thành List các vector trọng số thấp.
- **Triển khai:** 
  - Đẩy toàn bộ Vector Item $Q$ vào **FAISS index**. 
  - Đưa Vector User $P_u$ vào FAISS query khoảng cách Cosine hoặc L2 để lấy 200 items gần nhất. 

## Giai đoạn 2: RANKING (Xếp hạng tinh chỉnh)
- **Mục tiêu:** Lấy 200 items từ bước Recall, sử dụng các insight sâu (ngữ cảnh hiện tại, tính năng tổ hợp) để chấm lại điểm chính xác tuyệt đối (High precision).
- **Mô hình tiếp cận:** **LightGBM / XGBoost**
  - Xếp hạng như một bài toán Binary Classification (Dự đoán xác suất click CTR).
  - Kết quả xác suất này được sort descending ra Top 20 sản phẩm tốt nhất.

## Giai đoạn 3 (Bổ trợ): Fallback Strategy
- Giải quyết bài toán **Cold-start** khi User hoặc Item thụ động (chưa có interaction log).
- Fallback về hệ thống Recommend theo Item thịnh hành (Trending/Popularity) hoặc gán bằng Content-based Filtering (Dùng Metadata Vector tương đương nhau).