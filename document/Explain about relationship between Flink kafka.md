# Explain about relationship between Flink/kafka, Data warehouse và Feature Store

### 1. Tại sao 10 phút của Flink lại khác 10 phút của SQL/Batch?

Hãy nhìn vào ví dụ về đặc trưng: **"Số lần click trong 10 phút"**.

- **Cách làm của Batch (SQL trên Data Warehouse):**
Thông thường, Batch xử lý theo các khối cố định và không chồng lấn (Tumbling Windows).
    - Ví dụ: Khung 10:00 - 10:10, rồi đến 10:10 - 10:20.
    - Nếu người dùng click 3 lần lúc **10:09** và 3 lần lúc **10:11**.
    - **Kết quả Batch:** Bạn có hai bản ghi, mỗi bản ghi chỉ có 3 click. Mô hình AI học rằng: "Người dùng này chưa đạt ngưỡng 5 click".
- **Cách làm của Stream (Flink):**
Flink thường dùng cửa sổ trượt (Sliding Windows) hoặc cửa sổ theo phiên (Session Windows) được kích hoạt ngay khi có sự kiện mới.
    - Tại thời điểm **10:12**, Flink nhìn ngược lại 10 phút trước (từ 10:02 đến 10:12).
    - **Kết quả Stream:** Flink thấy tổng cộng 6 click (bao gồm cả lúc 10:09 và 10:11).
    - **Vấn đề:** Khi chạy thực tế, Flink gửi con số **"6"** cho AI. Nhưng AI trước đó chỉ được học trên dữ liệu Batch (toàn những con số **"3"**). AI sẽ bị bối rối vì nó chưa bao giờ thấy dữ liệu nào giống như vậy cho hành vi này.

---

### 2. "Late Data" (Dữ liệu đến muộn) - Lý do thuyết phục nhất

Đây là điểm mà Batch trong DW "đầu hàng" trước Stream của Flink, gây ra sự lệch lạc cực lớn.

- **Bối cảnh:** Người dùng đang đi tàu chui qua hầm, mất mạng. Họ click 5 lần lúc 10:00. Đến 10:15 họ mới có mạng lại, lúc này dữ liệu mới được gửi về server.
- **Trong Data Warehouse (Batch):** Job Spark chạy lúc 10:10 để tính toán dữ liệu khung 10:00 - 10:10. Lúc đó dữ liệu của người dùng này chưa về, nên kết quả là **0 click**. Sau đó dữ liệu mới về, nó nằm im trong kho và không được tính lại cho khung giờ đó nữa (trừ khi bạn chạy lại toàn bộ Batch job rất tốn kém).
- **Trong Flink (Stream):** Flink có cơ chế **Watermark**. Nó có thể đợi dữ liệu đến muộn. Khi dữ liệu lúc 10:00 cập bến vào lúc 10:15, Flink "nhớ" trạng thái cũ, mở lại cửa sổ 10:00 và cộng thêm 5 click này vào, sau đó cập nhật kết quả mới nhất cho AI.

**=> Kết luận:** Dữ liệu dùng để Train (Batch) báo là 0, nhưng dữ liệu dùng để chạy thực tế (Stream) báo là 5. AI sẽ đưa ra quyết định sai lầm.

---

### 3. Tại sao Flink lại gọi là "Stream" mà không phải "Small Batch"?

Bạn thắc mắc: *"Lấy dữ liệu trước đó 10 phút thì vẫn là batch chứ?"*. Câu trả lời nằm ở **Cơ chế kích hoạt (Triggering)**:

1. **Batch (Pull):** Hệ thống lập lịch (Scheduler) đúng 10:00 đi "kéo" dữ liệu về xử lý. Dữ liệu đứng yên, code đi tìm dữ liệu.
2. **Stream (Push):** Dữ liệu chảy đến đâu, Flink xử lý đến đó.
    - Mỗi khi có **1 click mới**, Flink ngay lập tức cập nhật lại con số tổng trong 10 phút qua.
    - Nó không đợi đến cuối giờ mới tính. Nó trả về một **dòng chảy các kết quả liên tục**. Đó là lý do gọi là Stream.

---

### 4. Feature Store giải quyết việc này như thế nào?

Feature Store không cố gắng biến Flink thành Batch. Nó giải quyết bằng cách **"Ép cả hai phải dùng chung một thước đo"**:

- **Lúc Train (Offline):** Thay vì để bạn tự viết SQL `GROUP BY` lộn xộn, Feature Store cung cấp một thư viện. Thư viện này mô phỏng lại chính xác cơ chế "cửa sổ trượt" và "xử lý dữ liệu muộn" của Flink trên dữ liệu lịch sử trong kho (Data Warehouse).
- **Lúc Chạy (Online):** Nó đảm bảo Flink sử dụng đúng công thức và tham số đó.

**Ví dụ cụ thể:**
Feature Store định nghĩa một Feature: `user_click_count_sliding_10m`.

- Nó bắt Spark (Offline) phải quét dữ liệu cũ và tính theo kiểu "trượt" từng phút một để tạo tập Train.
- Nó bắt Flink (Online) cũng phải dùng cửa sổ trượt tương tự.

Khi đó, AI được "học" trên các cửa sổ trượt (thấy con số 6 click) và khi chạy thực tế nó cũng "nhìn" thấy cửa sổ trượt (con số 6 click). Sự lệch lạc (Skew) biến mất.