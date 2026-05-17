# 7. Các Độ đo Đánh giá (Evaluation Metrics)

Hệ thống được thiết kế theo 2 giai đoạn (Recall và Rank), do đó việc đánh giá cũng được chia tách rõ ràng cho từng giai đoạn và đánh giá tổng thể trên góc độ nghiệp vụ.

## 1. Đánh giá Mô hình Offline (Offline Metrics)

### A. Đánh giá Giai đoạn Recall (Truy xuất ứng viên)
Ở bước này, mô hình (Matrix Factorization/Two-Tower kết hợp FAISS) trả về ~200 sản phẩm tiềm năng. Mục tiêu là không được "bỏ sót" sản phẩm mà người dùng thực sự muốn (độ phủ cao). Thứ tự chưa thực sự quan trọng ở bước này.
- **Hit Rate (HR@K):** Tỷ lệ phần trăm người dùng có sản phẩm họ thực sự tương tác nằm trong top K sản phẩm được Recall truy xuất ra (ví dụ: HR@200). Đơn giản là tính: *Có trúng hay không?*
- **Recall@K:** Thể hiện tỷ lệ các item tích cực của người dùng xuất hiện trong tập K ứng viên so với tổng số lượng item tích cực họ có. Đây là metric sống còn của bước này.

### B. Đánh giá Giai đoạn Ranking (Xếp hạng tinh chỉnh)
Ở bước này, mô hình (LightGBM/XGBoost) cố gắng xếp lại 200 items thành 20 items xuất sắc nhất. Thứ tự xếp hạng là yếu tố cực kỳ quan trọng (Sản phẩm người dùng thích nhất phải nằm ở Top 1, Top 2).
- **NDCG@K (Normalized Discounted Cumulative Gain):** Đo lường chất lượng xếp hạng dựa trên vị trí. Trọng số sẽ bị phạt nặng (discount) nếu sản phẩm phù hợp bị xếp ở vị trí thấp. (Đánh giá mạnh NDCG@10 hoặc NDCG@20).
- **MRR (Mean Reciprocal Rank):** Đánh giá việc mô hình đưa được sản phẩm tích cực đầu tiên lên vị trí số mấy. Tính bằng trung bình nghịch đảo của thứ hạng của item đúng đầu tiên.
- **Precision@K:** Tỷ lệ sản phẩm được đề xuất trong tập Top K thực sự được người dùng tương tác.

*Lưu ý:* Khi cấu hình train LightGBM, Loss function (Objective) có thể là Binary Logloss (Cross-entropy), nhưng hàm đánh giá (Eval Metric) BẮT BUỘC phải dùng NDCG.

## 2. Đánh giá Hệ thống & Nghiệp vụ (Online/Business Metrics)

Khi hệ thống triển khai A/B Testing trong thực tế (hoặc giả lập thực tế), nó phải đáp ứng các tiêu chuẩn sau:

### A. Độ đo Kinh doanh (Business Metrics)
- **CTR (Click-Through Rate):** Tỷ lệ nhấp chuột. Tổng số lượt User Click vào sản phẩm được gợi ý chia cho Tổng số lần sản phẩm được hiển thị (Impressions).
- **CR (Conversion Rate):** Tỷ lệ chuyển đổi. Bao nhiêu phần trăm trong số những cú Click đi đến quyết định Thêm vào giỏ (Add-to-cart) hoặc Mua hàng (Purchase).

### B. Độ đo Kỹ thuật Hệ thống (System Metrics)
- **Latency (Độ trễ phục vụ):** Tổng thời gian từ lúc quét FAISS + chạy LightGBM trả về kết quả phải **< 100ms** để đảm bảo trải nghiệm UX tại thời gian thực (đặc biệt khi chạy inference trên RTX 5060 hoặc CPU server).
- **Throughput / RPS (Requests Per Second):** Số lượng request gợi ý mà hệ thống có thể chịu tải trong 1 giây mà không bị sập.