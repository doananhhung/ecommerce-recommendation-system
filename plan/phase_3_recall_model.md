# Phase 3: Mô hình Truy xuất (Recall Model - Candidate Generation)

## Stage 3.1: Thử nghiệm Mô hình (Model Prototyping - `03_Model_Exp.ipynb`)
- [x] Xây dựng PyTorch `Dataset` và `DataLoader` cho dữ liệu tương tác User-Item.
- [x] Thiết kế kiến trúc Deep Learning (Matrix Factorization hoặc Two-Tower) bằng PyTorch (>=2.7).
- [x] Xác định hàm Loss Function phù hợp (VD: BPR Loss hoặc Contrastive Loss).
- [x] Train thử nghiệm mô hình trên GPU và trích xuất vector nhúng (Embeddings).

## Stage 3.2: Đưa vào Mã nguồn Chính (Productionizing Code)
- [x] Triển khai `src/recall_model/dataset.py`.
- [x] Triển khai `src/recall_model/model.py`.
- [x] Triển khai `src/recall_model/trainer.py` (chứa vòng lặp huấn luyện, optimizer).
- [x] Lưu trữ trọng số mô hình vào `models_store/recall_weights.pth`.
- [x] Lưu trữ vector nhúng của toàn bộ sản phẩm (Item Embeddings) ra file numpy `data/feature_store/item_embeddings.npy`.
