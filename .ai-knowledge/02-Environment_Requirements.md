# 2. Yêu cầu môi trường & Cấu hình (Environment Requirements)

## Thiết bị Phát triển (Target Hardware)
- **GPU:** NVIDIA RTX 5060.
- **Hệ điều hành:** Windows.

## Môi trường Phần mềm
- **Tần dụng sức mạnh GPU:** Project đặc biệt yêu cầu các framework tính toán song song, tối ưu hóa Cuda.
- **PyTorch:** Xây dựng & huấn luyện các mô hình Embedding/Deep Learning BẮT BUỘC sử dụng từ **PyTorch phiên bản 2.7 trở lên**.
  - *Sự cố nếu dùng version cũ:* Không tận dụng được các API tối ưu hóa cấp thấp trên dòng card RTX mới, thiếu hụt các hàm hỗ trợ Distributed Processing tối ưu có trong 2.7+.
- **LightGBM/XGBoost:** Bật chế độ `device="gpu"` để train Ranking model nhằm giảm thiểu thời gian.
- **FAISS:** Cài đặt bản `faiss-gpu` để tăng tốc độ index và query vector bằng CUDA.
- **UX:** Quản lý thư viện cho python 

