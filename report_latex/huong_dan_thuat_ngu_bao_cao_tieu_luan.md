# HƯỚNG DẪN NGÔN NGỮ VÀ THUẬT NGỮ CHO BÁO CÁO TIỂU LUẬN

## 1. Nguyên tắc xử lý thuật ngữ (Anh – Việt)

Đối với một bài báo cáo tiểu luận chuyên ngành Công nghệ thông tin hoặc Hệ thống gợi ý tại Việt Nam, việc cân bằng giữa tính học thuật (Academic) và tính kỹ thuật thực tế (Technical) là rất quan trọng.

Việc dịch toàn bộ thuật ngữ sang tiếng Việt sẽ làm báo cáo trở nên gượng ép và khó hiểu. Ngược lại, lạm dụng quá nhiều tiếng Anh sẽ làm mất đi tính học thuật của một bài tiểu luận viết bằng tiếng Việt.

Giải pháp phù hợp là sử dụng mô hình lai (Hybrid Approach) theo các nguyên tắc sau.

---

### Quy tắc 1: Dịch nghĩa kèm chú thích tiếng Anh ở lần xuất hiện đầu tiên

Đối với các khái niệm cốt lõi, nên sử dụng cụm từ tiếng Việt tương đương và chú thích thuật ngữ tiếng Anh trong ngoặc đơn ở lần đầu tiên xuất hiện.

Ví dụ:

> “Hệ thống gợi ý dựa trên kiến trúc hai giai đoạn (Two-Stage Recommender System)...”

Các lần sau có thể sử dụng linh hoạt giữa tiếng Việt hoặc tiếng Anh đã được chú thích.

#### Một số cặp thuật ngữ khuyến nghị

| Thuật ngữ tiếng Anh | Thuật ngữ khuyến nghị |
|---|---|
| Candidate Generation | Giai đoạn tạo ứng viên (Recall) |
| Ranking | Giai đoạn xếp hạng (Ranking) |
| Evaluation Metrics | Chỉ số đánh giá hiệu suất |

---

### Quy tắc 2: Giữ nguyên các thuật ngữ kỹ thuật phổ biến

Các thuật ngữ thuộc về:
- tên thuật toán,
- thư viện,
- framework,
- pipeline,
- hoặc khái niệm kỹ thuật chuyên sâu

nên được giữ nguyên bằng tiếng Anh.

#### Ví dụ nên giữ nguyên

- LightGBM
- Faiss Index
- Pipeline
- EDA (Exploratory Data Analysis)
- Session
- Sessionizer
- Smoke Testing
- Cold-start
- Overfitting

Lưu ý:
- Viết đúng định dạng danh từ riêng.
- Có thể in nghiêng với các khái niệm chuyên sâu như *pseudo-labeling*.

---

### Quy tắc 3: Việt hóa các từ đã có thuật ngữ chuẩn xác

Trong báo cáo khoa học, nhiều từ tiếng Anh quen thuộc vẫn nên được chuyển sang tiếng Việt để đảm bảo tính trang trọng.

| Không nên viết | Nên viết |
|---|---|
| Train mô hình | Huấn luyện mô hình |
| Test hệ thống | Kiểm thử hệ thống |
| Code | Mã nguồn |
| Run pipeline | Khởi chạy pipeline |
| Feature | Đặc trưng |

---

## 2. Phong cách hành văn (Tone & Style)

Báo cáo kỹ thuật yêu cầu:
- tính khách quan,
- tính chính xác,
- tính nhất quán.

### 2.1. Sử dụng ngôi kể khách quan (ngôi thứ ba)

Tránh sử dụng:
- “Tôi”
- “Chúng em”
- “Nhóm mình”

Nên sử dụng:
- “Nghiên cứu này”
- “Đề tài”
- “Hệ thống được đề xuất”
- “Nhóm tác giả”

#### Ví dụ

Không phù hợp:

> “Chúng tôi lựa chọn thuật toán LightGBM vì nó nhanh hơn.”

Phù hợp:

> “Thuật toán LightGBM được lựa chọn nhờ khả năng huấn luyện nhanh và hiệu quả cao trên dữ liệu dạng bảng.”

---

### 2.2. Tránh các từ ngữ cảm tính hoặc mơ hồ

Báo cáo kỹ thuật cần dựa trên:
- số liệu,
- chỉ số đánh giá,
- lập luận logic.

#### Tránh dùng

- “Rất mạnh”
- “Cực kỳ nhanh”
- “Khá tốt”
- “Tuyệt vời”

#### Nên thay bằng

- “Tăng X% chỉ số NDCG”
- “Tốc độ xử lý đạt X requests/giây”
- “Cải thiện đáng kể hiệu năng so với mô hình cơ sở”

---

### 2.3. Trình bày công thức toán học chuẩn xác bằng LaTeX

Tất cả:
- ký hiệu toán học,
- công thức,
- chỉ số đánh giá

nên được trình bày bằng LaTeX để đảm bảo tính học thuật và chuyên nghiệp.

#### Ví dụ

Nên viết:

```latex
NDCG@K
```

hoặc:

```latex
y = f(x)
```

thay vì viết thuần văn bản không định dạng.

---

## 3. Bảng đối chiếu thuật ngữ khuyến nghị cho dự án hệ thống gợi ý

| Thuật ngữ gốc (English) | Thuật ngữ khuyến nghị | Ghi chú |
|---|---|---|
| Two-Stage Architecture | Kiến trúc hai giai đoạn | Dịch nghĩa kèm tiếng Anh ở lần đầu |
| Recall (Candidate Generation) | Giai đoạn tạo ứng viên / mô hình Recall | Có thể dùng song song |
| Ranking | Giai đoạn xếp hạng |  |
| Feature Engineering | Kỹ nghệ trích xuất đặc trưng / Feature Engineering | Có thể giữ nguyên tiếng Anh |
| Pseudo-Labeling | Gán nhãn giả (*Pseudo-Labeling*) | Nên in nghiêng thuật ngữ |
| Sessionizer | Bộ phân đoạn phiên (Sessionizer) | Có thể giữ nguyên trong phần kỹ thuật |
| Serving Pipeline | Luồng phục vụ thời gian thực |  |
| Smoke Testing | Kiểm thử khói (*Smoke Testing*) | Giữ nguyên tiếng Anh trong ngoặc |
