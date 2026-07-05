# 🎭 Hệ thống Nhận dạng Cảm xúc Người học Online

> **Môn học:** Thị giác Máy tính — Đại học Bách khoa Hà Nội  
> **Mô hình:** EfficientNet-B0 (HSEmotion) + Fine-tuning  
> **Ngôn ngữ:** Python 3.8+ | PyTorch | OpenCV

---

## 📋 Mục lục

1. [Giới thiệu tổng quan](#1-giới-thiệu-tổng-quan)
2. [Các tập dữ liệu và model — AFEW, VGAF, AffectNet là gì?](#2-các-tập-dữ-liệu-và-model)
3. [Cấu trúc thư mục dự án](#3-cấu-trúc-thư-mục-dự-án)
4. [Sơ đồ hoạt động hệ thống](#4-sơ-đồ-hoạt-động-hệ-thống)
5. [Các chức năng và cách dùng](#5-các-chức-năng-và-cách-dùng)
6. [Giải thích thuật toán chi tiết](#6-giải-thích-thuật-toán-chi-tiết)
7. [Quy trình Fine-tuning — Giải thích từ đầu](#7-quy-trình-fine-tuning)
8. [Đánh giá mô hình](#8-đánh-giá-mô-hình)
9. [Cài đặt và chạy](#9-cài-đặt-và-chạy)
10. [Toàn bộ tham số dòng lệnh](#10-toàn-bộ-tham-số-dòng-lệnh)

---

## 1. Giới thiệu tổng quan

### 1.1. Mục tiêu dự án

Hệ thống nhận dạng **8 cảm xúc cơ bản** của người học qua webcam, video hoặc ảnh tĩnh theo thời gian thực, không cần kết nối internet hay API bên ngoài — toàn bộ xử lý chạy offline trên máy tính cá nhân.

**8 cảm xúc được nhận diện:**

| STT | Cảm xúc (VI) | Tên trong model | Biểu hiện điển hình |
|-----|-------------|-----------------|---------------------|
| 1 | Tức giận | `Anger` | Mày cau, môi mím chặt |
| 2 | Khinh thường | `Contempt` | Một bên khóe miệng nhếch lên |
| 3 | Ghê tởm | `Disgust` | Mũi nhăn, môi trên kéo lên |
| 4 | Sợ hãi | `Fear` | Mắt mở rộng, miệng há nhẹ |
| 5 | Vui vẻ | `Happiness` | Nụ cười, khóe mắt cong |
| 6 | Trung tính | `Neutral` | Không có biểu cảm rõ ràng |
| 7 | Buồn bã | `Sadness` | Khóe miệng kéo xuống, mắt rủ |
| 8 | Ngạc nhiên | `Surprise` | Mắt mở to, miệng há |

> **Lưu ý:** `Contempt` (khinh thường) rất hiếm trong dataset học trực tuyến nên thực tế hệ thống chủ yếu nhận diện 7 cảm xúc còn lại. Khi fine-tune, nhóm thường chỉ thu thập ảnh 7 class (bỏ Contempt).

### 1.2. Kiến trúc tổng thể — 2 giai đoạn

Hệ thống kết hợp **Computer Vision cổ điển** với **Deep Learning**:

```
┌────────────────────────────────────────────────────────────┐
│                  PIPELINE XỬ LÝ ẢNH                        │
│                                                            │
│  ĐẦU VÀO          GIAI ĐOẠN 1             GIAI ĐOẠN 2      │
│                  (Tìm khuôn mặt)        (Nhận diện CX)     │
│                                                            │
│  [Webcam]  ──►  [Haar Cascade]   ──►  [EfficientNet-B0]    │
│  [Video ]       Thuật toán cũ         Mạng học sâu         │
│  [Ảnh   ]       ~1ms/frame           ~50–100ms/lần         │
└────────────────────────────────────────────────────────────┘
```

**Tại sao chia 2 giai đoạn?**

Nếu dùng EfficientNet trực tiếp trên toàn bộ frame 640×480 → rất chậm vì phải quét toàn bộ ảnh lớn. Thay vào đó:
- **Giai đoạn 1:** Haar Cascade nhanh (~1ms) xác định tọa độ (x, y, w, h) của từng khuôn mặt
- **Giai đoạn 2:** EfficientNet chỉ cần xử lý vùng ảnh nhỏ đã cắt ra (224×224px) → nhanh hơn nhiều

### 1.3. Phù hợp với sinh viên năm 2 không?

**Hoàn toàn phù hợp.** Lý do:

| Yêu cầu | Thực tế dự án |
|---------|---------------|
| Hiểu Python cơ bản | ✅ Code rõ ràng, có comment tiếng Việt đầy đủ |
| Cần biết Deep Learning | ❌ Không cần — model đã được train sẵn, nhóm chỉ fine-tune phần nhỏ |
| Phần "tự làm" | ✅ Thu thập dataset, fine-tune, đánh giá, viết báo cáo |
| Thư viện phức tạp | ❌ HSEmotion che giấu toàn bộ PyTorch phức tạp, dùng như hàm đơn giản |
| GPU đắt tiền | ❌ Không cần, chạy tốt trên CPU laptop |

Phần quan trọng nhất mà nhóm **tự làm** là:
1. Thu thập dataset (chụp ảnh khuôn mặt nhóm + tìm ảnh online)
2. Chạy fine-tuning và quan sát kết quả
3. Phân tích confusion matrix để giải thích tại sao model nhầm lẫn
4. So sánh độ chính xác trước/sau fine-tuning trong báo cáo

---

## 2. Các tập dữ liệu và model

### 2.1. Ba checkpoint model là gì?

Thư viện HSEmotion cung cấp **3 file model đã được train sẵn** (pre-trained), mỗi file được huấn luyện trên bộ dữ liệu khác nhau:

#### `enet_b0_8_best_afew` — **Dùng mặc định trong dự án này**

- Huấn luyện trên **AffectNet** (dữ liệu chính) + **AFEW** (dữ liệu bổ sung)
- Tốc độ nhanh nhất trong 3 model
- Phù hợp nhất cho môi trường video thực tế (webcam)

#### `enet_b0_8_best_vgaf`

- Huấn luyện trên **AffectNet** + **VGAF** (dữ liệu bổ sung)  
- Tốc độ ngang model trên, độ chính xác nhỉnh hơn một chút

#### `enet_b2_8`

- Chỉ dùng **AffectNet**
- Model lớn hơn (EfficientNet-B2 thay vì B0) → chính xác hơn nhưng chậm hơn ~2x
- Nên dùng khi phân tích video sau khi quay (không cần real-time)

### 2.2. AffectNet là gì?

**AffectNet** là tập dữ liệu ảnh khuôn mặt lớn nhất thế giới dành cho nhận diện cảm xúc:

- **~450,000 ảnh** khuôn mặt có nhãn cảm xúc
- Thu thập bằng cách tìm kiếm Google/Bing với từ khóa kiểu "happy face", "angry face"...
- Đa dạng: nhiều dân tộc, độ tuổi, ánh sáng, góc chụp
- Mỗi ảnh được gắn nhãn bởi nhiều người → giảm sai sót

**Ý nghĩa:** Model đã "học" từ 450,000 khuôn mặt nên hiểu rất tốt biểu cảm khuôn mặt nói chung. Fine-tuning trên dataset nhỏ của nhóm giúp nó học thêm đặc điểm riêng.

### 2.3. AFEW là gì?

**AFEW (Acted Facial Expressions in the Wild)** là tập dữ liệu gồm các đoạn video ngắn trích từ phim điện ảnh, diễn viên biểu diễn cảm xúc trong bối cảnh tự nhiên (không phải ảnh chụp tĩnh). Việc train thêm trên AFEW giúp model quen với ảnh chất lượng webcam thấp, chuyển động, ánh sáng thay đổi.

### 2.4. VGAF là gì?

**VGAF (Video-level Group AFfect)** là tập dữ liệu video về cảm xúc nhóm người — phù hợp hơn cho tình huống lớp học online có nhiều người trong khung hình.

### 2.5. Có cần dùng cả 3 model không?

**Không cần.** Dùng **1 model duy nhất** là đủ cho cả dự án.

Ba cái đó không phải 3 mô hình khác nhau về kiến trúc, mà chỉ là **cùng 1 mạng EfficientNet-B0 nhưng được huấn luyện trên dữ liệu khác nhau** — giống như 3 bạn học cùng một sách giáo khoa nhưng luyện ở 3 bộ đề thi khác nhau:

```
enet_b0_8_best_afew → luyện đề AffectNet + AFEW (đề video thực tế)
enet_b0_8_best_vgaf → luyện đề AffectNet + VGAF (đề nhóm người)
enet_b2_8           → luyện đề AffectNet (học kỹ hơn, lâu hơn, chính xác hơn)
```

**Chọn model nào cho dự án này?**

```
Tình huống                           Model nên chọn
──────────────────────────────────────────────────────────
Demo real-time webcam (mặc định)     enet_b0_8_best_afew
Ưu tiên độ chính xác hơn tốc độ     enet_b0_8_best_vgaf
Phân tích video đã quay sẵn         enet_b2_8
```

**Gợi ý câu giải thích cho báo cáo** (copy và chỉnh tên nhóm):

> *"Nhóm chọn `enet_b0_8_best_afew` vì model này được huấn luyện bổ sung trên tập AFEW gồm các đoạn video thực tế, phù hợp hơn với môi trường webcam học online có chuyển động và thay đổi ánh sáng liên tục. Ngoài ra đây là model nhẹ nhất trong 3 lựa chọn (EfficientNet-B0), đảm bảo chạy real-time trên CPU thông thường mà không cần GPU — phù hợp với điều kiện máy tính sinh viên."*

---

## 3. Cấu trúc thư mục dự án

```
KT/
│
├── main.py                  ← ĐIỂM VÀO CHÍNH — chạy file này
│                              Có 5 chế độ: webcam, image, video,
│                              finetune, evaluate
│
├── requirements.txt         ← Danh sách thư viện cần cài (pip install)
│
│
├── src/                     ← Thư mục chứa code xử lý
│   │
│   ├── capture.py           ← Quản lý mở/đọc/đóng camera
│   │                          Xử lý hiển thị cửa sổ OpenCV
│   │
│   ├── emotion_engine.py    ← MODULE CỐT LÕI
│   │                          - Haar Cascade detect khuôn mặt
│   │                          - CLAHE tăng tương phản ảnh
│   │                          - EfficientNet predict cảm xúc
│   │                          - Temporal Smoothing làm mượt kết quả
│   │                          - Hỗ trợ load fine-tuned weights
│   │
│   ├── fine_tuner.py        ← Fine-tune model trên dataset nhỏ
│   │                          - Đóng băng phần lớn mạng neural
│   │                          - Huấn luyện lại các lớp cuối
│   │                          - Lưu model tốt nhất vào models/
│   │
│   ├── image_predictor.py   ← Nhận diện cảm xúc từ ảnh tĩnh
│   │                          Vẽ bounding box + thanh xác suất
│   │
│   ├── evaluator.py         ← So sánh model gốc vs fine-tuned
│   │                          Tính Accuracy, Precision, Recall, F1
│   │                          Vẽ confusion matrix
│   │
│   ├── analytics.py         ← (Cũ) Vẽ biểu đồ thống kê phiên học
│   └── database.py          ← (Cũ) Lưu trữ SQLite
│
│
├── dataset/                 ← Ảnh tự thu thập để fine-tune
│   ├── README.md            ← Hướng dẫn thu thập ảnh
│   ├── Anger/               ← ≥30 ảnh khuôn mặt tức giận
│   ├── Disgust/             ← ≥30 ảnh ghê tởm
│   ├── Fear/                ← ≥30 ảnh sợ hãi
│   ├── Happiness/           ← ≥30 ảnh vui vẻ
│   ├── Neutral/             ← ≥30 ảnh trung tính
│   ├── Sadness/             ← ≥30 ảnh buồn
│   └── Surprise/            ← ≥30 ảnh ngạc nhiên
│
├── models/                  ← Lưu weights sau fine-tune
│   └── finetuned.pt         ← (Tự tạo ra khi chạy finetune)
│
├── data/                    ← Dữ liệu khi chạy
│   └── screenshot_*.jpg     ← Ảnh chụp màn hình khi nhấn 's'
│
└── reports/                 ← Kết quả đánh giá mô hình
    ├── confusion_matrix.png ← Ma trận nhầm lẫn
    └── training_curve.png   ← Biểu đồ loss/accuracy theo epoch
```

---

## 4. Sơ đồ hoạt động hệ thống

### 4.1. Vòng lặp xử lý frame (Webcam & Video)

```
╔═══════════════════════════════════════════════════════════════╗
║                    VÒNG LẶP CHÍNH                             ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  ┌─────────────────────────────────────────────────────┐      ║
║  │ 1. Đọc frame mới từ camera/video                    │      ║
║  └──────────────────────┬──────────────────────────────┘      ║
║                         │                                     ║
║                         ▼                                     ║
║  ┌─── Frame số chẵn? ───────────────────────────────────┐     ║
║  │                                                       │     ║
║  │  CÓ (frame chẵn)           KHÔNG (frame lẻ)          │     ║
║  │      │                         │                     │     ║
║  │      ▼                         ▼                     │     ║
║  │  Chạy AI detect            Dùng kết quả              │     ║
║  │  khuôn mặt +               lần trước                 │     ║
║  │  nhận diện CX              (không chạy AI)            │     ║
║  │      │                         │                     │     ║
║  └──────┴────────────┬────────────┘─────────────────────┘     ║
║                      │                                        ║
║                      ▼                                        ║
║  ┌─────────────────────────────────────────────────────┐      ║
║  │ 2. Vẽ bounding box + nhãn CX lên frame             │      ║
║  └──────────────────────┬──────────────────────────────┘      ║
║                         │                                     ║
║                         ▼                                     ║
║  ┌─────────────────────────────────────────────────────┐      ║
║  │ 3. Hiển thị / Lưu vào video output                 │      ║
║  └──────────────────────┬──────────────────────────────┘      ║
║                         │                                     ║
║                         ▼                                     ║
║  ┌─────────────────────────────────────────────────────┐      ║
║  │ 4. Xử lý phím bấm                                  │      ║
║  │    q/ESC → thoát                                   │      ║
║  │    s → lưu ảnh                                     │      ║
║  │    r → reset buffer                                │      ║
║  └──────────────────────┬──────────────────────────────┘      ║
║                         │                                     ║
║              ┌──────────┴──────────┐                          ║
║              ▼                     ▼                          ║
║           [Thoát]         [Đọc frame tiếp]                    ║
╚═══════════════════════════════════════════════════════════════╝
```

### 4.2. Chi tiết xử lý AI cho mỗi khuôn mặt

```
Khuôn mặt thô (vùng ảnh BGR)
        │
        ▼
┌───────────────────────────────────────────────┐
│ BƯỚC 1 — CLAHE Preprocessing                  │
│ BGR → Grayscale → CLAHE → RGB                 │
│ (Cân bằng sáng để làm rõ đặc trưng biểu cảm) │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│ BƯỚC 2 — EfficientNet Forward Pass            │
│ Input: ảnh RGB 224×224px                      │
│ Output: 8 số (xác suất mỗi cảm xúc)          │
│ Ví dụ: [0.05, 0.02, 0.03, 0.04,              │
│          0.72, 0.08, 0.04, 0.02]              │
│          Ang   Con   Dis   Fea                │
│                Hap   Neu   Sad   Sur          │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│ BƯỚC 3 — Temporal Smoothing                   │
│ Thêm kết quả vào buffer 10 frame             │
│ Tính trung bình → giảm nhiễu, mượt hơn       │
│                                               │
│ Frame 1: [0.05, 0.72, ...]                   │
│ Frame 2: [0.08, 0.68, ...]  → trung bình     │
│ ...                         → [0.06, 0.70, .]│
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│ BƯỚC 4 — Adaptive Threshold                   │
│ Nếu xác suất cao nhất < ngưỡng               │
│ → mặc định là Neutral                         │
│ Ngưỡng riêng: Happiness≥0.50, Sadness≥0.30...│
└───────────────────────┬───────────────────────┘
                        │
                        ▼
              [Cảm xúc cuối cùng + độ tin cậy]
```

### 4.3. Luồng Fine-tuning

```
┌─────────────────────────────────────────────────────────────┐
│                  QUY TRÌNH FINE-TUNING                       │
└─────────────────────────────────────────────────────────────┘

CHUẨN BỊ             HUẤN LUYỆN              KẾT QUẢ
───────────          ─────────────────         ────────

[Thu thập ảnh]       [Mỗi epoch:]
     │                  Train set              [models/
     │                  (70%) đưa vào    ──►   finetuned.pt]
     ▼                  model                        │
[dataset/                 │                          │
 Anger/      ]            ▼                          ▼
 Happiness/  ]        Tính sai số            [Dùng để predict]
 Neutral/    ]            │                  [python main.py
 ...]                     ▼                   --weights
     │                Cập nhật            models/finetuned.pt]
     │                trọng số
     ▼                    │
[Chia tự động]            ▼
 70% Train            Val set (15%)
 15% Val        ──►   kiểm tra độ
 15% Test             chính xác
                           │
                           ▼
                      Lưu nếu tốt hơn
                      lần trước
```

---

## 5. Các chức năng và cách dùng

### Chế độ 1: Webcam real-time

Mở webcam và nhận diện cảm xúc liên tục. Đây là chế độ demo chính.

```bash
# Dùng model gốc
python main.py --mode webcam

# Dùng model đã fine-tune
python main.py --mode webcam --weights models/finetuned.pt

# Dùng webcam ID 1 (nếu có nhiều camera)
python main.py --mode webcam --camera 1
```

**Phím bấm khi chạy:**
- `q` hoặc `ESC` → Thoát chương trình
- `s` → Chụp ảnh màn hình, lưu vào thư mục `data/`
- `r` → Xóa bộ nhớ đệm Temporal Smoothing (làm mới nhận diện)

---

### Chế độ 2: Nhận diện từ ảnh tĩnh

Đưa 1 file ảnh vào, hệ thống phát hiện khuôn mặt và nhận diện cảm xúc.

```bash
# Nhận diện và hiển thị
python main.py --mode image --input anh_khuon_mat.jpg

# Nhận diện và lưu ảnh kết quả
python main.py --mode image --input anh_khuon_mat.jpg --output ket_qua.jpg

# Dùng model fine-tuned
python main.py --mode image --input anh_khuon_mat.jpg --weights models/finetuned.pt
```

Kết quả hiển thị:
- Bounding box màu sắc theo cảm xúc (xanh lá=vui, đỏ=tức giận...)
- Nhãn cảm xúc + phần trăm độ tin cậy
- Thanh xác suất 8 cảm xúc ở góc phải màn hình

---

### Chế độ 3: Nhận diện từ video

Đọc file video, xử lý từng frame và hiển thị nhận diện cảm xúc theo thời gian thực.

```bash
# Xem video với nhận diện
python main.py --mode video --input video_hoc.mp4

# Xem và lưu video kết quả (có overlay)
python main.py --mode video --input video_hoc.mp4 --output ket_qua.mp4
```

**Định dạng video hỗ trợ:** `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`

**Phím bấm:**
- `q` / `ESC` → Dừng
- `SPACE` → Tạm dừng / Tiếp tục
- `s` → Chụp ảnh frame hiện tại

Thanh màu xanh ở dưới màn hình thể hiện tiến trình video (Frame X/Y, thời gian).

---

### Chế độ 4: Fine-tune model

Huấn luyện lại phần cuối của model trên dataset nhỏ do nhóm tự thu thập.

```bash
# Fine-tune với 15 epoch (mặc định)
python main.py --mode finetune --dataset dataset/

# Chỉnh số epoch và learning rate
python main.py --mode finetune --dataset dataset/ --epochs 20 --lr 5e-5

# Chọn model nền khác
python main.py --mode finetune --dataset dataset/ --model enet_b0_8_best_vgaf
```

Khi chạy sẽ thấy output từng epoch:
```
Epoch [ 1/15] Train Loss: 1.4231 Acc: 48.3% | Val Loss: 1.3102 Acc: 52.1%
Epoch [ 2/15] Train Loss: 1.1823 Acc: 61.4% | Val Loss: 1.0945 Acc: 64.7%  ✓ Saved!
Epoch [ 3/15] Train Loss: 0.9234 Acc: 69.1% | Val Loss: 0.8901 Acc: 71.2%  ✓ Saved!
...
```

---

### Chế độ 5: Đánh giá mô hình

So sánh độ chính xác giữa model gốc và model fine-tuned trên test set.

```bash
python main.py --mode evaluate --dataset dataset/ --weights models/finetuned.pt
```

---

<!-- ### Thu thập dataset — Đã bỏ, không còn dùng

Script `collect_data.py` từng dùng để chụp ảnh từ webcam đã được loại bỏ khỏi dự án.
Thay vào đó, dataset được chuẩn bị thủ công theo cấu trúc dataset/<ClassName>/ảnh.jpg.

--- -->

## 6. Giải thích thuật toán chi tiết

### 6.1. Haar Cascade — Phát hiện khuôn mặt

**File:** `src/emotion_engine.py`, hàm `detect_faces()`

Haar Cascade là thuật toán phát hiện đối tượng được Viola và Jones đề xuất năm 2001. Dù đã hơn 20 năm tuổi nhưng vẫn được dùng vì tốc độ rất cao.

**Nguyên lý hoạt động:**

*Bước 1 — Haar features (đặc trưng Haar):*

Thuật toán chia ảnh thành các vùng hình chữ nhật và tính hiệu độ sáng giữa hai vùng liền kề:

```
Ví dụ đặc trưng mắt:         Ví dụ đặc trưng mũi:
┌────────┬────────┐           ┌──────────────────┐
│ TRẮNG  │  ĐEN   │           │      TRẮNG        │
│(trán)  │ (mắt) │           ├────────┬──────────┤
└────────┴────────┘           │ ĐEN(mũi)│  TRẮNG  │
                              └────────┴──────────┘
Giá trị = tổng(trắng) - tổng(đen)
Mắt thường tối hơn trán → giá trị âm → đặc trưng mắt khớp!
```

*Bước 2 — Integral Image (ảnh tích phân):*

Để tính tổng pixel trong bất kỳ hình chữ nhật nào chỉ trong O(1) thời gian (thay vì phải duyệt từng pixel):

```
Ảnh gốc:          Integral Image:
1  2  3            1   3   6
4  5  6     →      5  12  21
7  8  9            12  27  45

Tổng vùng bất kỳ = tra bảng 4 góc → O(1)
```

*Bước 3 — AdaBoost Cascade (phân loại tầng):*

Thay vì dùng 1 bộ phân loại phức tạp, dùng nhiều bộ phân loại đơn giản xếp thành tầng. Vùng ảnh không phải mặt bị loại ngay từ tầng đầu → tiết kiệm 99% thời gian:

```
Tầng 1: (2 đặc trưng)   → Loại ngay 60% vùng không phải mặt
Tầng 2: (10 đặc trưng)  → Loại thêm 80% còn lại
Tầng 3: (25 đặc trưng)  → Loại tiếp
...
Tầng N: (200 đặc trưng) → Kết luận cuối: CÓ mặt hay KHÔNG
```

**Code thực tế:**

```python
faces = self.face_classifier.detectMultiScale(
    gray,
    scaleFactor=1.1,   # Mỗi bước thu nhỏ ảnh 10% để detect mặt nhiều kích thước
    minNeighbors=5,    # Vùng phải được 5 vùng lân cận xác nhận → giảm báo sai
    minSize=(30, 30)   # Bỏ qua mặt nhỏ hơn 30×30px (quá xa, không đáng tin)
)
# Trả về: [(x1,y1,w1,h1), (x2,y2,w2,h2), ...]  — tọa độ từng khuôn mặt
```

---

### 6.2. CLAHE — Cân bằng sáng thích nghi

**File:** `src/emotion_engine.py`, hàm `_preprocess_face()`

**Vấn đề:** Webcam trong phòng học thường có ánh sáng không đều — cửa sổ chiều sáng từ một phía, đèn huỳnh quang từ trên. Ảnh khuôn mặt có thể quá tối hoặc quá sáng cục bộ → model nhận diện kém.

**CLAHE (Contrast Limited Adaptive Histogram Equalization)** giải quyết bằng cách cân bằng sáng theo từng vùng nhỏ:

```
Ảnh gốc: khuôn mặt nửa sáng nửa tối do ánh sáng cửa sổ
         ┌─────────────────────────────┐
         │ SÁNG (phía cửa) │ TỐI      │
         │   Nhìn rõ nét   │ Mờ nhạt  │
         └─────────────────────────────┘

Sau CLAHE:
         ┌─────────────────────────────┐
         │ Cân bằng tương phản         │
         │ Cả hai phía đều rõ nét     │
         │ Nếp nhăn, đường viền rõ    │
         └─────────────────────────────┘
```

**Tại sao "Adaptive"?** Histogram Equalization thông thường cân bằng trên toàn ảnh → có thể làm mất chi tiết. CLAHE chia ảnh thành **lưới 8×8 tile** và cân bằng riêng từng tile → giữ được chi tiết cục bộ.

**"Contrast Limited"** nghĩa là giới hạn mức khuếch đại (`clipLimit=2.0`) để tránh khuếch đại nhiễu ảnh quá mức.

```python
self.clahe = cv2.createCLAHE(
    clipLimit=2.0,        # Giới hạn khuếch đại để tránh noise
    tileGridSize=(8, 8)   # Chia ảnh thành lưới 8×8 tile
)
gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
equalized = self.clahe.apply(gray)
processed = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)  # HSEmotion cần RGB
```

---

### 6.3. EfficientNet — Mạng nhận diện cảm xúc

**EfficientNet** (Google, 2019) là mạng neural tích chập (CNN) hiệu quả nhất hiện tại trong nhóm lightweight model.

**Ý tưởng Compound Scaling:**

Các CNN trước đây muốn tăng độ chính xác chỉ tăng 1 trong 3 yếu tố:
- Tăng độ sâu (thêm lớp) → VGG, ResNet
- Tăng độ rộng (thêm filter) → WideResNet
- Tăng độ phân giải đầu vào → tốn RAM

EfficientNet tăng **đồng thời cả 3** theo tỉ lệ cân bằng → hiệu quả hơn nhiều:

```
                Tham số   ImageNet Acc
ResNet-50       25.6M     76.2%
EfficientNet-B0  5.3M     77.1%   ← Ít hơn 5 lần, chính xác hơn!
EfficientNet-B2  9.2M     80.1%
```

**Kiến trúc tổng quát của EfficientNet-B0:**

```
Ảnh đầu vào: 224×224×3 (RGB)
      │
      ▼
Stem Conv: 32 filter 3×3
      │
      ▼
MBConv Block 1 (×1):  16 filter   → học đặc trưng cơ bản (cạnh, góc)
MBConv Block 2 (×2):  24 filter   → kết hợp đặc trưng
MBConv Block 3 (×2):  40 filter   → hình dạng đơn giản
MBConv Block 4 (×3):  80 filter   → vùng khuôn mặt
MBConv Block 5 (×3): 112 filter   → mắt, miệng, mũi
MBConv Block 6 (×4): 192 filter   → biểu cảm phức tạp  ← Fine-tune
MBConv Block 7 (×1): 320 filter   → đặc trưng cảm xúc  ← Fine-tune
      │
      ▼
Global Average Pooling → vector 1280 chiều
      │
      ▼
Classifier (Fully Connected) → 8 số (1 cho mỗi cảm xúc)
      │
      ▼
Softmax → 8 xác suất, tổng = 1.0
```

**MBConv Block là gì?**  
Mobile Inverted Bottleneck Convolution — đây là block tích chập hiệu quả được phát triển cho thiết bị di động, dùng **Depthwise Separable Convolution** để giảm phép tính xuống ~9 lần so với Conv thông thường mà không mất nhiều độ chính xác.

---

### 6.4. Per-Face Temporal Smoothing — Làm mượt kết quả

**File:** `src/emotion_engine.py`

**Vấn đề:** Ngay cả khi bạn giữ nguyên biểu cảm, EfficientNet có thể trả ra kết quả hơi khác nhau mỗi frame do noise ảnh nhỏ, thay đổi ánh sáng, hơi di chuyển. Ví dụ:

```
Frame 1: Happy (72%)
Frame 2: Happy (68%)      → Nhãn ổn định
Frame 3: Neutral (38%)    ← Nhiễu! Thực ra vẫn Happy
Frame 4: Happy (74%)      → Nhãn ổn định
```

Frame 3 sẽ gây hiệu ứng nhấp nháy khó chịu.

**Giải pháp — Buffer 10 frame:**

```python
# Mỗi khuôn mặt có một buffer riêng (deque = hàng đợi vòng)
if face_key not in self.face_buffers:
    self.face_buffers[face_key] = deque(maxlen=10)  # Giữ 10 frame gần nhất

buf = self.face_buffers[face_key]
buf.append(scores)                  # Thêm kết quả mới

avg_scores = np.mean(buf, axis=0)  # Tính trung bình 10 frame
```

Sau khi lấy trung bình:
```
Frame 1-10: [0.72, 0.68, 0.38, 0.74, 0.71, 0.69, 0.70, 0.73, 0.72, 0.70]
Trung bình:  0.677 → vẫn Happy (67.7%) → không bị nhấp nháy!
```

**Nhận diện từng khuôn mặt riêng (Per-Face Key):**

Với nhiều người trong khung hình, mỗi khuôn mặt cần buffer RIÊNG. Nếu dùng chung 1 buffer, dữ liệu mặt A và mặt B sẽ trộn lẫn → kết quả sai.

```python
def _get_face_key(self, x, y, w, h):
    cx = ((x + w//2) // 60) * 60   # Tọa độ tâm, làm tròn theo lưới 60px
    cy = ((y + h//2) // 60) * 60
    return (cx, cy)
```

Lưới 60px có nghĩa là nếu khuôn mặt dịch chuyển dưới 60px (do đầu nhúc nhích nhẹ) thì vẫn nhận ra là cùng 1 người → buffer không bị reset, vẫn mượt.

---

### 6.5. Adaptive Threshold — Ngưỡng thích nghi

**File:** `src/emotion_engine.py`

Thay vì dùng ngưỡng cố định 0.5 cho tất cả cảm xúc, mỗi cảm xúc có ngưỡng riêng dựa trên đặc điểm thực tế:

```python
self.thresholds = {
    'Happiness': 0.50,  # Cần xác suất cao mới kết luận vui
                        # → tránh nhận nhầm khi người học mỉm cười lịch sự

    'Sadness':   0.30,  # Ngưỡng thấp → nhạy với buồn
                        # → người học ít khi biểu lộ buồn rõ ràng
                        #   nhưng hệ thống vẫn nên phát hiện

    'Surprise':  0.35,  # Ngưỡng thấp → bắt được khoảnh khắc
                        # → ngạc nhiên xảy ra rất nhanh (< 0.5 giây)

    'Anger':     0.40,
    'Neutral':   0.35   # Default khi không có cảm xúc rõ ràng
}

# Nếu xác suất cao nhất < ngưỡng → kết quả là Neutral
if max_conf < threshold:
    final_emotion = 'Neutral'
```

---

### 6.6. Skip-Frame — Cân bằng FPS và AI

**File:** `main.py`, hàm `run_webcam()` và `run_video()`

EfficientNet mất ~50–100ms để xử lý 1 frame. Nếu mỗi frame đều chạy AI:
- FPS thực tế = 1000ms / 100ms = **10 FPS** → hình ảnh giật rõ

Giải pháp: Chỉ chạy AI mỗi 2 frame, frame bị bỏ qua dùng kết quả cũ:

```python
frame_count += 1

if frame_count % 2 != 0:      # Frame lẻ: bỏ qua AI
    display = draw(frame, last_results)   # Vẽ kết quả CŨ lên frame MỚI
else:                           # Frame chẵn: chạy AI
    last_results = engine.process_frame(frame)
    display = draw(frame, last_results)
```

Kết quả:
- Camera đọc 30 frame/giây
- AI chạy mỗi 2 frame = 15 lần/giây
- Hiển thị vẫn 30 FPS (frame lẻ dùng kết quả cũ, không nhấp nháy)

---

## 7. Quy trình Fine-tuning

### 7.1. Fine-tuning là gì? (Giải thích đơn giản)

Hãy tưởng tượng model EfficientNet đã được học từ 450,000 khuôn mặt người — nó hiểu biểu cảm khuôn mặt rất tốt. Tuy nhiên, nó học từ ảnh Internet (nhiều dân tộc, điều kiện ánh sáng khác nhau). Dataset của nhóm là khuôn mặt sinh viên Việt Nam trong phòng học → có thể có đặc điểm riêng.

**Fine-tuning = Giữ kiến thức cũ + Học thêm từ data mới**

Cụ thể:
- **Đóng băng (freeze) 80% lớp đầu:** Giữ nguyên, không cho phép thay đổi. Các lớp này đã học được đặc trưng chung rất tốt (cạnh, đường nét, hình dạng cơ bản).
- **Mở đóng băng (unfreeze) 2 lớp cuối + classifier:** Cho phép cập nhật theo dataset của nhóm. Các lớp cuối chịu trách nhiệm "quyết định" cảm xúc dựa trên đặc trưng → cần điều chỉnh theo data mới.

**Ví dụ trực quan:**

```
Giống như học tiếng Anh có sẵn (frozen),
rồi học thêm tiếng lóng/accent địa phương (unfrozen)
→ Không cần học lại từ đầu!
```

### 7.2. Tại sao không train toàn bộ từ đầu?

Dataset nhỏ (~200 ảnh) mà train 5.3M tham số → **Overfitting**: model "học thuộc lòng" training set, không tổng quát hóa được.

```
Train accuracy: 99%   (học thuộc)
Test accuracy:  45%   (không nhớ gì cả khi gặp ảnh mới)
```

Với freeze: Chỉ train ~500K tham số, dataset 200 ảnh là đủ.

### 7.3. Chiến lược Freeze/Unfreeze

```
EfficientNet-B0 — 7 nhóm MBConv Block:

Nhóm 1 (Stem + Block 1–5): ████████████░  ĐÓNG BĂNG (80%)
  → Học đặc trưng cơ bản: cạnh, góc, hình dạng
  → Không thay đổi — đã tốt rồi

Nhóm 2 (Block 6–7):       ░░░░░░░░████   MỞ ĐÓNG BĂNG
  → Học đặc trưng cảm xúc cụ thể
  → Điều chỉnh theo dataset mới

Classifier (Linear 320→8):░░░░░░░████    MỞ ĐÓNG BĂNG
  → Quyết định cuối cùng
  → Quan trọng nhất cần điều chỉnh

Tổng tham số frozen:   ~4.8M (80%) — không thay đổi
Tổng tham số trainable: ~0.5M (20%) — học từ dataset nhỏ
```

### 7.4. Cách model học — Training Loop

Mỗi lần lặp (epoch), model trải qua 2 pha:

**Pha Train:**

```python
for batch in train_loader:           # Lấy từng batch 16 ảnh
    output = model(images)           # Forward pass: dự đoán cảm xúc
    loss = CrossEntropy(output, labels)  # Tính sai số
    loss.backward()                  # Backpropagation: tính gradient
    optimizer.step()                 # Cập nhật trọng số theo gradient
    optimizer.zero_grad()            # Xóa gradient để batch tiếp theo
```

**CrossEntropyLoss** đo độ sai của dự đoán:
```
Nhãn thật: Happy (index 4)
Dự đoán:   [0.05, 0.02, 0.03, 0.04, 0.72, 0.08, 0.04, 0.02]
Loss = -log(0.72) = 0.33   ← thấp = dự đoán tốt

Nếu đoán sai:
Dự đoán:   [0.05, 0.02, 0.03, 0.04, 0.15, 0.60, 0.09, 0.02]
Loss = -log(0.15) = 1.90   ← cao = dự đoán tệ
```

**Pha Validation:**

Sau mỗi epoch, chạy model trên validation set (không cập nhật trọng số) để kiểm tra có bị overfitting không. Nếu val accuracy tốt hơn lần trước → lưu checkpoint.

**Learning Rate Scheduler:**

```
Epoch 1–5:  lr = 0.0001       ← học nhanh
Epoch 6–10: lr = 0.00005      ← học chậm hơn để tinh chỉnh
Epoch 11–15: lr = 0.000025    ← học rất chậm, ổn định
```

Giảm dần learning rate giúp model hội tụ ổn định, tránh dao động.

### 7.5. Dataset Split — Tại sao cần chia 3?

Giả sử nhóm thu thập được **210 ảnh** (30 ảnh × 7 class). Chia ngay từ đầu thành 3 phần tách biệt hoàn toàn:

```
Dataset tổng: 210 ảnh (30 ảnh × 7 class)
      │
      ├── Train set (70% = 147 ảnh)
      │   └── Model ĐƯỢC PHÉP học từ đây
      │       Trọng số được cập nhật dựa trên bộ này
      │       Mỗi epoch duyệt qua 147 ảnh này nhiều lần
      │
      ├── Validation set (15% = 32 ảnh)
      │   └── Model KHÔNG được train trên đây
      │       Sau mỗi epoch: chạy model trên 32 ảnh này
      │       để kiểm tra "học xong có thực sự tốt hơn không"
      │       → Dùng để quyết định có lưu checkpoint không
      │
      └── Test set (15% = 31 ảnh)
          └── Cất kỹ, KHÔNG ĐƯỢC XEM cho đến khi train xong hẳn
              Chỉ dùng DUY NHẤT 1 LẦN ở cuối
              → Đây là con số accuracy chính thức trong báo cáo
```

**Hình dung đơn giản:**
- **Train set** = sách giáo khoa để học
- **Validation set** = đề luyện tập để tự kiểm tra
- **Test set** = đề thi thật — chỉ làm 1 lần, không được xem trước

**Có cần thay test set mỗi lần test không?**

Không. Test set chỉ có **1 bộ duy nhất, cố định**. Nhưng quy tắc là **không được dùng kết quả test set để ra quyết định gì**:

```
❌ SAI — Dùng test set nhiều lần để điều chỉnh:

  Train xong lần 1 → test → 67% → "Chưa đủ, tăng epochs lên"
  Train lại lần 2  → test → 71% → "Vẫn thấp, đổi learning rate"
  Train lại lần 3  → test → 74% → "OK rồi, báo cáo 74%"

  → Test set đã bị "rò rỉ" vào quá trình ra quyết định
  → Model gián tiếp được tối ưu CHO test set
  → Số 74% không còn đáng tin, bị phồng lên

✅ ĐÚNG — Dùng val set để điều chỉnh, test set chỉ dùng 1 lần:

  Train + điều chỉnh (chỉ nhìn val set)
       │
       │ Khi hài lòng với val accuracy rồi mới...
       ▼
  Chạy test set → 71% → ĐÂY LÀ CON SỐ BÁO CÁO, không chỉnh gì nữa
```

**Với BTL của nhóm — thực tế chỉ cần làm thế này:**

```bash
# ~~Bước thu thập ảnh đã bỏ, dùng dataset có sẵn~~

# 2. Fine-tune (code tự chia train/val/test, seed cố định = 42
#    → mỗi lần chạy đều chia ra đúng bộ đó, không bị xáo trộn)
python main.py --mode finetune --dataset dataset/

# 3. Đánh giá 1 lần duy nhất → lấy số liệu báo cáo
python main.py --mode evaluate --dataset dataset/ --weights models/finetuned.pt
# → Ra accuracy + confusion matrix → chép vào báo cáo
```

Cô giáo chủ yếu quan tâm 3 điều:
1. Nhóm **giải thích được** tại sao chia 3 phần trong báo cáo
2. **So sánh được** accuracy model gốc vs fine-tuned
3. **Phân tích** confusion matrix: class nào bị nhầm, tại sao

---

## 8. Đánh giá mô hình

### 8.1. Accuracy (Độ chính xác)

Công thức đơn giản nhất:

```
Accuracy = Số ảnh đoán đúng / Tổng số ảnh × 100%

Ví dụ: 26 đúng / 31 ảnh test = 83.9%
```

**Giới hạn của Accuracy:** Nếu dataset mất cân bằng (ví dụ 80% ảnh là Neutral, 20% là các cảm xúc khác), model chỉ cần đoán Neutral cho tất cả cũng đạt 80% accuracy — nhưng thực ra vô dụng!

→ Cần thêm Precision, Recall, F1-score.

### 8.2. Precision, Recall, F1-score

Lấy ví dụ đánh giá riêng cho class **Anger**:

```
Kết quả thực tế:   [Anger, Anger, Happy, Angry, Neutral, Anger, Anger]
Model dự đoán:     [Anger, Happy, Happy, Angry, Anger,   Anger, Anger]

TP (True Positive):  Đoán Anger, thực sự là Anger     = 4
FP (False Positive): Đoán Anger, thực ra KHÔNG phải   = 1 (Neutral bị nhầm)
FN (False Negative): Thực ra Anger, đoán thành khác   = 1 (bị nhầm thành Happy)

Precision = TP / (TP + FP) = 4 / 5 = 80%
→ Trong số những lần model nói "Anger", 80% là đúng

Recall    = TP / (TP + FN) = 4 / 5 = 80%
→ Trong tất cả ảnh Anger thật, model tìm ra được 80%

F1-score  = 2 × (P × R) / (P + R) = 80%
→ Trung bình hài hòa giữa Precision và Recall
```

### 8.3. Confusion Matrix

Ma trận hiển thị chi tiết model nhầm cái gì với cái gì:

```
Ví dụ (31 ảnh test, 7 class):

              ← MODEL DỰ ĐOÁN LÀ →
              Ang Dis Fea Hap Neu Sad Sur
THỰC  Anger  [ 7   0   1   0   0   1   0 ]  → 7/9 đúng (Anger)
TẾ    Disgust[ 0   8   0   0   1   0   0 ]  → 8/9 đúng
      Fear   [ 0   0   5   0   0   0   3 ]  → 5/8 đúng (hay nhầm Surprise!)
      Happy  [ 0   0   0   9   0   0   0 ]  → 9/9 đúng
      Neutral[ 0   1   0   1   7   0   0 ]  → 7/9 đúng
      Sadness[ 1   0   0   0   2   6   0 ]  → 6/9 đúng
      Surprise[0   0   2   0   0   0   7 ]  → 7/9 đúng

Đường chéo = đoán ĐÚNG (số càng lớn càng tốt)
Ngoài đường chéo = đoán SAI

Nhận xét: Fear và Surprise hay nhầm lẫn nhau
→ Hợp lý! Cả hai đều có: mắt mở to, miệng há
→ Cần thu thập thêm ảnh Fear để phân biệt rõ hơn
```

Confusion matrix được vẽ dưới dạng heatmap (màu tối = nhiều ảnh), lưu tự động vào `reports/confusion_matrix.png`.

### 8.4. Chạy đánh giá

```bash
python main.py --mode evaluate --dataset dataset/ --weights models/finetuned.pt
```

Output mẫu:
```
====================================================
     ĐÁNH GIÁ MÔ HÌNH EMOTION DETECTION
====================================================

📂 Test set: 31 ảnh | 7 classes

  Model gốc  (HSEmotion pretrained): 71.0%
  Model fine-tuned:                  83.9%  (+12.9%)

📋 Classification Report (Model Fine-tuned):
              precision  recall  f1-score  support
       Anger     0.88    0.78      0.82        9
     Disgust     0.89    0.89      0.89        9
        Fear     0.71    0.63      0.67        8
       Happy     0.90    1.00      0.95        9
     Neutral     0.70    0.78      0.74        9
     Sadness     0.86    0.67      0.75        9
    Surprise     0.70    0.78      0.74        9
```

---

## 9. Cài đặt và chạy

### 9.1. Yêu cầu

- Python 3.8 trở lên
- RAM tối thiểu 4GB
- Webcam (cho chế độ webcam)
- Không cần GPU

### 9.2. Cài đặt

```bash
# Bước 1: Tải code về máy
git clone https://github.com/ncthangtt/khaithac.git
cd khaithac

# Bước 2: Tạo môi trường ảo Python
python -m venv venv

# Bước 3: Kích hoạt môi trường ảo
.\venv\Scripts\activate       # Windows
source venv/bin/activate      # Linux / macOS

# Bước 4: Cài thư viện
pip install -r requirements.txt
```

> **Lần đầu chạy:** Hệ thống tự tải model (~25MB) từ internet về `C:\Users\<tên>\\.hsemotion\`

### 9.3. Luồng làm việc đề xuất cho nhóm

```bash
# ── BƯỚC 1: ~~Thu thập ảnh~~ (đã bỏ, dataset chuẩn bị thủ công)

# ── BƯỚC 2: Fine-tune (chạy 1 lần, mất 15-30 phút) ──
python main.py --mode finetune --dataset dataset/ --epochs 15

# ── BƯỚC 3: Đánh giá và lấy số liệu cho báo cáo ──
python main.py --mode evaluate --dataset dataset/ --weights models/finetuned.pt
# Kết quả lưu vào reports/

# ── BƯỚC 4: Demo ──
python main.py --mode webcam --weights models/finetuned.pt
python main.py --mode image --input face.jpg --weights models/finetuned.pt
python main.py --mode video --input video.mp4 --output result.mp4
```

---

## 10. Toàn bộ tham số dòng lệnh

| Tham số | Giá trị | Mặc định | Dùng ở mode |
|---------|---------|---------|-------------|
| `--mode` | `webcam` / `image` / `video` / `finetune` / `evaluate` | `webcam` | Tất cả |
| `--model` | `enet_b0_8_best_afew` / `enet_b0_8_best_vgaf` / `enet_b2_8` | `enet_b0_8_best_afew` | Tất cả |
| `--weights` | Đường dẫn file `.pt` | `None` (dùng model gốc) | webcam, image, video |
| `--camera` | Số nguyên (0, 1, 2...) | `0` | webcam |
| `--input` | Đường dẫn file ảnh/video | `None` | image, video |
| `--output` | Đường dẫn file kết quả | `None` | image, video |
| `--dataset` | Đường dẫn thư mục dataset | `None` | finetune, evaluate |
| `--epochs` | Số nguyên | `15` | finetune |
| `--lr` | Số thực (ví dụ: `1e-4`) | `0.0001` | finetune |
| `--batch-size` | Số nguyên | `16` | finetune, evaluate |
| `--save-weights` | Đường dẫn lưu `.pt` | `models/finetuned.pt` | finetune |

---

*Đại học Bách khoa Hà Nội — Môn Thị giác Máy tính*  
*Nhóm sinh viên thực hiện: [Tên nhóm]*
