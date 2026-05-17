# 11. Tầm quan trọng của Timestamp (Tại sao dữ liệu bắt buộc phải có thời gian?)

Trong hệ thống Recommender System, đặc biệt là hệ thống có áp dụng tư duy Streaming hoặc mô phỏng luồng Streaming thông qua Batch (như project của chúng ta), cột **`Timestamp` (Dấu thời gian)** đóng vai trò như "xương sống". Nếu thiếu nó, toàn bộ các logic kỹ thuật tiên tiến nhất của thuật toán sẽ đổ vỡ. 

Dưới đây là 4 lý do cốt lõi giải thích tại sao `Timestamp` lại mang tính sát thủ và không thể thay thế:

## 1. Phục vụ chia phiên (Sessionization) - Hiểu ý định tức thời
Hành vi của người dùng thay đổi liên tục. Sáng họ tìm mua "Laptop", tối họ lại tìm mua "Giày thể thao". 
- Nếu không có thời gian, máy học (AI) sẽ gộp chung Laptop và Giày thể thao thành một khối "Sở thích của User A", dẫn đến gợi ý lẫn lộn, không đúng trọng tâm hiện tại.
- **Có Timestamp:** Máy tính sẽ Gom cụm (Group) được hành vi. Ví dụ, thiết lập cấu hình cửa sổ là 30 phút (Session Window). Các click diễn ra cách nhau dưới 30 phút sẽ được tính là một Phiên (Session) "Đang tìm mua Laptop". Vượt qua ngưỡng 30 phút, hệ thống tự hiểu người dùng đã bắt đầu một ý định mới ở một Session mới.

## 2. Tạo nhãn giả (Pseudo-labeling) chính xác
Bài toán gợi ý là bài toán One-Class (chúng ta chỉ thấy cái người dùng Click, không có nút Dislike để biết họ ghét cái gì). Do đó, ta phải "đoán" họ ghét cái gì bằng cách tạo **Nhãn giả (Pseudo-negative Labels = 0)**.
- **Nếu không có Timestamp:** Ta không biết lúc nào họ nhìn thấy sản phẩm. Từ đó phải lấy ngẫu nhiên các sản phẩm trong kho (Random Negative Sampling) để gán nhãn 0. Rất thiếu chính xác!
- **Có Timestamp và Session:** Hệ thống sẽ khoanh vùng được: "Trong phiên truy cập lúc 10h sáng, User A lướt qua (Impression) 10 cái Laptop, nhưng chỉ nán lại Click vào 2 cái, 8 cái còn lại bị ngó lơ". Nhờ có Timestamp định hình phiên, ta tự tin lấy 8 chiếc Laptop bị ngó lơ đó gắn nhãn `0`, và 2 chiếc phân tích sâu gắn nhãn `1`. Đây là tín hiệu học (Signal) cực kì chất lượng cho mô hình LightGBM (Giai đoạn Ranking).

## 3. Tránh rò rỉ dữ liệu tương lai (Data Leakage / Time Travel)
Quy tắc cấm kị số 1 trong Học máy là dùng dữ liệu Tương lai để dự đoán... Quá khứ.
- **Có Timestamp:** Tính toán **Point-in-time join** trong Feature Store. Nghĩa là lúc huấn luyện AI để dự đoán xem User A có mua Điện thoại vào ngày `Ngày 15` hay không, hệ thống sẽ sử dụng chỉ tổng số lượng Click của User A tính ĐẾN `Ngày 14` để làm Data Train.
- Nếu không có Timestamp để chốt chặn (Chỉ có tổng lịch sử), mô hình sẽ lấy Tổng số lượt Click của User trong cả tháng (chứa lọt cả dữ liệu từ `Ngày 16, 17`) vào để tính xác suất cho `Ngày 15`. Lúc này độ chính xác khi Train lên đến 99%, nhưng mang ra thực tế (khi tương lai chưa xảy ra) mô hình sẽ chạy cực kỳ tệ.

## 4. Mô hình hóa chuỗi hành vi (Sequential Modeling)
Một số mô hình Deep Learning trong Giai đoạn Recall (Ví dụ: DIN - Deep Interest Network, BERT4Rec hoặc RNN) quyết định việc gợi ý dựa trên TRẬT TỰ hành vi. 
- *Ví dụ:* User Mua một chiếc Điện thoại -> sau đó tìm mua Ốp lưng -> Hệ thống tự đoán bước tiếp theo họ cần Tai nghe.
- Nếu mất đi `Timestamp`, hệ thống sẽ xáo trộn lịch sử của họ thành một Túi (Bag-of-words), không phân biệt được trước hay sau, phá hỏng thuật toán Sequential Recommender.
