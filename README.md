# He thong Nhan dang Cam xuc Nguoi hoc Online

> **Mon hoc:** Khai thac thong tin da phuong tien — Dai hoc Bach khoa Ha Noi  
> **Mo hinh:** EfficientNet-B0 (HSEmotion) + Fine-tuning  
> **Ngon ngu:** Python 3.8+ | PyTorch | OpenCV

---

## 1. Gioi thieu

### 1.1. Muc tieu

He thong nhan dang **6 cam xuc co ban** cua nguoi hoc qua webcam, video hoac anh tinh theo thoi gian thuc, offline hoan toan.

**6 cam xuc:**

| Cam xuc | Mo ta |
|---------|-------|
| `Anger` | Tuc gian |
| `Disgust` | Ghe tom |
| `Happiness` | Vui ve |
| `Neutral` | Trung tinh |
| `Sadness` | Buon ba |
| `Surprise` | Ngac nhien |

### 1.2. Kien truc tong the

He thong chi co **1 file duy nhat** — `main.py` — gom 3 phan:

```
main.py
├── Phan 1: Constants + Imports (~15 dong)
├── Phan 2: EmotionEngine class (~70 dong)
│   - Haar Cascade phat hien mat
│   - HSEmotion predict cam xuc
│   - Temporal Smoothing + Adaptive Threshold
├── Phan 3: Cac mode functions (~200 dong)
│   - webcam mode
│   - image mode
│   - video mode
│   - finetune mode
│   - evaluate mode
└── Phan 4: main() + argparse (~40 dong)
```

### 1.3. Co che hoat dong

```
Dau vao (webcam/image/video)
        │
        ▼
Detect mat (Haar Cascade)
        │
        ▼
Crop tung khuon mat ≥48px
        │
        ▼
CLAHE preprocessing (Gray → CLAHE → RGB)
        │
        ▼
HSEmotion EfficientNet → 6 xac suat
        │
        ▼
Temporal Smoothing (trung binh 10 frame)
        │
        ▼
Adaptive Threshold → Emotion cuoi cung
        │
        ▼
Ve bounding box + hien thi
```

---

## 2. Cau truc thu muc

```
tucode/
│
├── main.py                   ← FILE DUY NHAT — chua toan bo code
├── README.md                 ← Tai lieu (file nay)
├── GIAI_THICH_DU_AN.html    ← Tai lieu giai thich
├── requirements.txt          ← Thu vien can cai
│
├── dataset/                  ← Bo du lieu
│   ├── train/                ← Tap train & val (366 anh)
│   │   ├── Anger/           61 anh
│   │   ├── Disgust/         61 anh
│   │   ├── Happiness/       61 anh
│   │   ├── Neutral/         61 anh
│   │   ├── Sadness/         61 anh
│   │   └── Surprise/        61 anh
│   │
│   └── test_dung1landuynhat/ ← Tap test doc lap (150 anh)
│       ├── Anger/           25 anh
│       ├── Disgust/         25 anh
│       ├── Happiness/       25 anh
│       ├── Neutral/         25 anh
│       ├── Sadness/         25 anh
│       └── Surprise/        25 anh
│
├── models/                   ← .pt weights sau khi fine-tune
│   └── finetuned.pt          (tu tao ra)
│
└── reports/                  ← Confusion matrix + training curve
    ├── confusion_matrix.png
    └── training_curve.png
```

---

## 3. Cai dat va chay

### 3.1. Yeu cau

- Python 3.8+
- RAM toi thieu 4GB
- Khong can GPU

### 3.2. Cai dat

```bash
# Tao moi truong ao (khuyen nghi)
python -m venv venv
.\venv\Scripts\activate     # Windows

# Cai thu vien
pip install -r requirements.txt
```

> **Lan dau chay:** HSEmotion tu dong tai model (~25MB) ve `C:\Users\<ten>\.hsemotion\`

### 3.3. Cac che do

```bash
# Webcam real-time
python main.py --mode webcam

# Webcam dung model fine-tuned
python main.py --mode webcam --weights models/finetuned.pt

# Nhan dien tu anh
python main.py --mode image --input anh.jpg
python main.py --mode image --input anh.jpg --output ketqua.jpg

# Nhan dien tu video
python main.py --mode video --input video.mp4
python main.py --mode video --input video.mp4 --output result.mp4

# Fine-tune
python main.py --mode finetune --dataset dataset/train --epochs 15

# Danh gia
python main.py --mode evaluate --dataset dataset/test_dung1landuynhat --weights models/finetuned.pt
```

### 3.4. Phim tat (webcam/video)

| Phim | Chuc nang |
|------|-----------|
| `q` / `ESC` | Thoat |
| `s` | Chup anh |
| `SPACE` | Tam dung (video) |

---

## 4. Giai thuat chinh

| Thanh phan | Mo ta |
|------------|-------|
| **Haar Cascade** | Phat hien mat ~1ms/frame, dung CPU |
| **CLAHE** | Can bang sang cuc bo, lam ro net mat |
| **EfficientNet-B0** | Neural network ~5.3M tham so, nhanh + chinh xac |
| **Temporal Smoothing** | Deque 10 frame, tinh trung binh → manh, ko bi nhap nhay |
| **Adaptive Threshold** | Nguong rieng cho tung emotion |

---

## 5. Fine-tuning

### 5.1. Chien luoc

1. Tai model HSEmotion pre-trained (hoc tu 450k anh AffectNet)
2. Dong bang toan bo backbone (giu kien thuc cu)
3. Mo 2 blocks cuoi + classifier (hoc them data moi)
4. Train/Val/Test split 70/15/15

### 5.2. Chay

```bash
python main.py --mode finetune --dataset dataset/train --epochs 15
```

Ket qua luu vao `models/finetuned.pt`.

### 5.3. Danh gia

```bash
python main.py --mode evaluate --dataset dataset/test_dung1landuynhat --weights models/finetuned.pt
```

So sanh accuracy model goc vs fine-tuned, xuat confusion matrix + classification report vao `reports/`.

---

## 6. Tham so dong lenh

| Tham so | Gia tri | Mac dinh | Mode |
|---------|---------|----------|------|
| `--mode` | webcam/image/video/finetune/evaluate | webcam | Tat ca |
| `--model` | enet_b0_8_best_afew / enet_b0_8_best_vgaf / enet_b2_8 | enet_b0_8_best_afew | Tat ca |
| `--weights` | duong dan .pt | None | webcam, image, video |
| `--camera` | so int | 0 | webcam |
| `--input` | duong dan file | None | image, video |
| `--output` | duong dan file | None | image, video |
| `--dataset` | duong dan thu muc | None | finetune, evaluate |
| `--epochs` | so int | 15 | finetune |
| `--lr` | so thuc | 0.0001 | finetune |
| `--batch-size` | so int | 16 | finetune, evaluate |
| `--save-weights` | duong dan .pt | models/finetuned.pt | finetune |

---

*Dai hoc Bach khoa Ha Noi — Mon Khai thac thong tin da phuong tien*
