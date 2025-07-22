# Hướng dẫn cài đặt & sử dụng ALFWorld

## 1. Yêu cầu hệ thống
- Python >= 3.7 (khuyến nghị dùng 3.8 hoặc 3.9)
- pip, virtualenv hoặc venv
- RAM >= 8GB (khuyến nghị 16GB nếu train nhiều)
- (Tùy chọn) GPU + CUDA nếu muốn train nhanh hơn
- Hệ điều hành: Ubuntu, WSL2, hoặc Linux (TextWorld chạy tốt trên Windows Subsystem for Linux)

## 2. Tạo và kích hoạt môi trường ảo
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Cài đặt thư viện phụ thuộc
```bash
pip install --upgrade pip
pip install -r requirements-full.txt
```
Nếu gặp lỗi về torch, hãy cài bản phù hợp với máy:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 4. Chuẩn bị dữ liệu ALFWorld
- Tải/copy bộ dữ liệu ALFWorld (các thư mục json_2.1.1, logic, ...) vào đúng vị trí (theo biến ALFWORLD_DATA hoặc alfworld/data/)
- Nếu chưa có, có thể dùng script hoặc hướng dẫn trong README.md gốc để tải dữ liệu mẫu.

## 5. Cấu hình môi trường
- Sửa file `configs/base_config.yaml` hoặc `configs/eval_config.yaml` nếu muốn thay đổi batch size, learning rate, đường dẫn dữ liệu, ...
- Đảm bảo các trường như `save_path`, `use_cuda`, `training_method` đúng với nhu cầu.

## 6. Train agent (ví dụ PPO)
```bash
python scripts/train_ppo.py --config configs/base_config.yaml
```
- Checkpoint sẽ được lưu trong thư mục `training/`.
- Quá trình train sẽ in log reward, loss, số bước, checkpoint định kỳ.

## 7. Đánh giá agent đã train
```bash
python scripts/evaluate_checkpoint.py --config configs/eval_config.yaml --checkpoint training/test_ep10000.pt
```
- Kết quả sẽ được ghi vào file log hoặc eval_results.csv.

## 8. Chạy thử nghiệm (TextWorld)
- Chơi bằng tay:
```bash
python scripts/alfworld-play-tw $ALFWORLD_DATA/json_2.1.1/train/ten_nhiem_vu/trial_xxx/
```
- Agent tự động chơi (ví dụ PPO):
```bash
python app_ppo.py --checkpoint training/test_ep10000.pt --problem $ALFWORLD_DATA/json_2.1.1/train/ten_nhiem_vu/trial_xxx/
```
- Hoặc dùng web UI (nếu đã setup backend):
  - Chạy backend: `python alfworld_ppo_api.py`
  - Mở file `static_alfworld_ppo.html` trên trình duyệt

## 9. Một số lỗi thường gặp & cách khắc phục
- **Lỗi thiếu thư viện:** Đảm bảo đã cài đúng requirements-full.txt, kiểm tra lại torch, textworld, gym, ...
- **Lỗi bus error, core dumped:** Thường do thiếu RAM hoặc checkpoint bị lỗi, thử với checkpoint khác hoặc giảm batch size.
- **Lỗi CUDA:** Nếu không có GPU, hãy đặt `use_cuda: false` trong config.
- **Lỗi không tìm thấy dữ liệu:** Kiểm tra biến ALFWORLD_DATA, đường dẫn dữ liệu trong config.

## 10. Tài liệu tham khảo
- [ALFWorld GitHub](https://github.com/alfworld/alfworld)
- [ALFWorld Paper](https://arxiv.org/abs/2010.03768)
- [TextWorld](https://aka.ms/textworld)

---
Nếu gặp khó khăn, hãy đọc kỹ log lỗi, kiểm tra lại các bước, hoặc hỏi trên GitHub Issues của dự án! 