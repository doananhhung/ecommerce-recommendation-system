# Danh sách Công việc (TODO List) - Cải tiến Đặc trưng Phiên & Recall Hai Kênh

- `[x]` 1. Cấu hình các đặc trưng mới và tham số recall kênh trong [config.py](file:///D:/programing/project/EDA_project/src/config.py)
- `[x]` 2. Cập nhật đặc trưng phiên lũy tiến trong [featurizer.py](file:///D:/programing/project/EDA_project/src/data_pipeline/featurizer.py)
  - `[x]` Cập nhật hàm `add_point_in_time_features` cho `user_session_interaction_count` và `item_session_popularity`
  - `[x]` Cập nhật hàm `extract_item_features` để merge tổng số phiên chứa sản phẩm vào Serving Feature Store
- `[x]` 3. Triển khai Triệu hồi Hai Kênh và chỉ thị kênh gợi ý trong [pipeline.py](file:///D:/programing/project/EDA_project/src/serving/pipeline.py)
  - `[x]` Bổ sung tham số `session_items` vào hàm gợi ý
  - `[x]` Triển khai Kênh 2: Lấy embeddings các item trong phiên -> Tính vector trung bình -> Truy vấn FAISS
  - `[x]` Gộp ứng viên, loại trùng, và tạo hai cột chỉ thị kênh `recalled_by_long_term` và `recalled_by_session`
  - `[x]` Tính toán động `user_session_interaction_count` từ `session_items` và điền khuyết
- `[x]` 4. Hỗ trợ truyền tham số lịch sử phiên qua Query Parameters tại [main_serve.py](file:///D:/programing/project/EDA_project/main_serve.py)
- `[x]` 5. Xác minh toàn bộ hệ thống bằng cách chạy lại Pipeline, Train và khởi động API Server
  - *Lưu ý:* Mã nguồn và logic tích hợp đã được triển khai hoàn chỉnh. Do môi trường Windows có lỗi tương thích cấp hệ thống về đường dẫn terminal trong sandbox, việc chạy các lệnh shell tự động bị chặn. Bạn có thể tự kích hoạt quy trình này trên thiết bị cục bộ bằng các câu lệnh trong tài liệu hướng dẫn.
