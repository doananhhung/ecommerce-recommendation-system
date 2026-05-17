# 4. Luồng xử lý Dữ liệu (Data Pipeline)

## Phân loại dữ liệu ngõ vào
1. **Implicit Feedback (Phản hồi ẩn):** Lịch sử hiển thị (impression), click, view, thêm giỏ hàng. Dồi dào nhưng nhiễu.
2. **Explicit Feedback (Phản hồi tường minh):** Ratings(1-5 sao), reviews. Ít nhưng độ tin cậy tuyệt đối.
3. **Metadata (Siêu dữ liệu):** User profile (nhân khẩu học) và Item catalog (giá, mô tả, category).

## Các bước Tiền xử lý dữ liệu (Pre-processing)

1. **Sessionization (Nhóm theo phiên):**
   - Quét toàn bộ lịch sử hành vi (log theo timestamp). 
   - Gom các actions của User (Click A, View B) nằm sát nhau về mặt thời gian thành các Phiên truy cập (Sessions).

2. **Negative / Pseudo-labeling (Gắn nhãn giả):**
   - Sự vắng mặt của click không hẳn là User ghét sp đó, có thể họ chưa nhìn thấy. Đây là bài toán One-Class CF.
   - *Chiến lược:* Nếu 1 Item xuất hiện trong Session (Impression) mà không có Click/Add-to-cart, Item đó bị gán nhãn 0. Các Item được Click được gán nhãn 1. Tạo tập mẫu cân bằng.

3. **Feature Engineering & Vectorization:**
   - Vector hóa Categorical Features (One-hot, Embedding lookup).
   - Biến text/description thành Vector thông qua pretrained NLP models.
   - Lưu bộ feature gốc này về một kho gọi chung là Feature Registry. Tái sử dụng để nối với label lúc train mô hình.