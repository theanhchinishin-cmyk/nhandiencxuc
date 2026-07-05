# Hướng dẫn tạo Dataset

Thư mục này chứa ảnh khuôn mặt để fine-tune mô hình nhận dạng cảm xúc.

## Cấu trúc thư mục

```
dataset/
├── Anger/        ← ảnh khuôn mặt tức giận
├── Disgust/      ← ảnh khuôn mặt ghê tởm
├── Fear/         ← ảnh khuôn mặt sợ hãi
├── Happiness/    ← ảnh khuôn mặt vui vẻ
├── Neutral/      ← ảnh khuôn mặt trung tính
├── Sadness/      ← ảnh khuôn mặt buồn
└── Surprise/     ← ảnh khuôn mặt ngạc nhiên
```

> ⚠️ **Tên thư mục phải viết đúng chính xác** như trên (có phân biệt hoa thường).

---

## Cách 1: Thu thập bằng webcam (khuyến nghị)

```bash
# ~~Script collect_data.py đã bỏ — dùng dataset thủ công~~
```

- Dataset được đặt thủ công vào thư mục dataset/<ClassName>/ảnh.jpg
- Các class hợp lệ: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
- Nhấn **n / p** để chuyển class tiếp/trước
- Nhấn **q** để thoát

---

## Cách 2: Tải ảnh từ internet

Các nguồn ảnh miễn phí phù hợp:
- [FER2013 (Kaggle)](https://www.kaggle.com/datasets/msambare/fer2013)
- [AffectNet subset](https://huggingface.co/datasets/Piro17/affectnet)
- Google Images (tìm theo từ khóa "happy face", "angry face", ...)

Sau khi tải về, đặt ảnh vào đúng thư mục class.

---

## Yêu cầu tối thiểu

| Class | Số ảnh tối thiểu | Khuyến nghị |
|-------|-----------------|-------------|
| Anger | 20 | 40–50 |
| Disgust | 20 | 40–50 |
| Fear | 20 | 40–50 |
| Happiness | 20 | 40–50 |
| Neutral | 20 | 40–50 |
| Sadness | 20 | 40–50 |
| Surprise | 20 | 40–50 |

**Tổng tối thiểu:** 140 ảnh | **Khuyến nghị:** 280–350 ảnh

---

## Bước tiếp theo sau khi có dataset

```bash
# 1. Fine-tune model
python main.py --mode finetune --dataset dataset/ --epochs 15

# 2. Đánh giá kết quả
python main.py --mode evaluate --dataset dataset/

# 3. Dùng model fine-tuned
python main.py --mode webcam --weights models/finetuned.pt
python main.py --mode image --input face.jpg --weights models/finetuned.pt
```
