# Hệ thống Nhận dạng Cảm xúc Người học Online

> **Môn học:** Khai thác thông tin đa phương tiện — Đại học Bách khoa Hà Nội  
> **Mô hình:** EfficientNet-B0 (HSEmotion) với cơ chế remap 8 sang 6 lớp  
> **Ngôn ngữ:** Python 3.8+ | PyTorch | OpenCV  
> **Độ chính xác thực nghiệm:** **81.33%** trên tập dữ liệu Đông Nam Á độc lập  

---

## 1. Giới thiệu

Hệ thống nhận dạng **6 cảm xúc cơ bản** của người học qua webcam, video hoặc ảnh tĩnh theo thời gian thực, hoạt động offline hoàn toàn.

**6 cảm xúc nhận diện:**
- `Anger` (Tức giận)
- `Disgust` (Ghê tởm)
- `Happiness` (Vui vẻ)
- `Neutral` (Bình thường / Trung tính)
- `Sadness` (Buồn bã)
- `Surprise` (Ngạc nhiên)

### 1.1. Kiến trúc luồng xử lý (Pipeline)
```
Đầu vào (webcam/image/video)
        │
        ▼
Detect mặt (Haar Cascade)
        │
        ▼
Crop từng khuôn mặt ≥48px
        │
        ▼
CLAHE preprocessing (Gray → CLAHE → RGB)
        │
        ▼
HSEmotion EfficientNet → 8 xác suất
        │
        ▼
Remap sang 6 lớp & Tái chuẩn hóa Softmax
        │
        ▼
Temporal Smoothing (trung bình trượt 10 frame)
        │
        ▼
Adaptive Threshold (Ngưỡng thích nghi riêng từng cảm xúc)
        │
        ▼
Vẽ bounding box màu sắc + Hiển thị
```

---

## 2. Cấu trúc thư mục

```
nhandiencxuc/
│
├── main.py                   ← FILE DUY NHẤT — chứa toàn bộ code chạy
├── README.md                 ← Hài liệu hướng dẫn sử dụng (file này)
├── GIAI_THICH_DU_AN.html    ← Trang web giải thích chi tiết đồ án
├── requirements.txt          ← Danh sách các thư viện cần cài đặt
│
└── reports/                  ← Báo cáo thống kê cảm xúc phiên học (tự xuất)
    └── emotion_report.png    ← Biểu đồ tròn và timeline phân bố cảm xúc
```

---

## 3. Cài đặt và Chạy hệ thống

### 3.1. Yêu cầu hệ thống
- Python 3.8+
- RAM tối thiểu 4GB
- Chạy mượt mà trực tiếp trên **CPU** máy tính cá nhân phổ thông.

### 3.2. Cài đặt thư viện
Mở PowerShell hoặc Command Prompt tại thư mục dự án và chạy:
```bash
# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
.\venv\Scripts\activate     # Windows

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```
> **Lưu ý lần đầu chạy:** Mô hình `enet_b0_8_best_afew` (~25MB) sẽ được tự động tải về thư mục `C:\Users\<tên_user>\.hsemotion\`.

### 3.3. Các lệnh thực thi chính

#### 1. Chạy Webcam thời gian thực (Mặc định)
Hệ thống sẽ mở camera, nhận diện cảm xúc chính diện và tự động lưu biểu đồ báo cáo khi bạn thoát:
```bash
python main.py --mode webcam
```

#### 2. Nhận diện từ file Ảnh tĩnh
Đọc ảnh, vẽ bounding box cảm xúc và xuất ra ảnh kết quả:
```bash
python main.py --mode image --input duong_dan_anh.jpg [--output ket_qua.jpg]
```

#### 3. Nhận diện từ file Video
Đọc luồng video, hiển thị nhận diện kèm thanh tiến trình bên dưới, ghi nhận lịch sử và xuất Dashboard báo cáo khi hết video hoặc khi ngắt:
```bash
python main.py --mode video --input duong_dan_video.mp4 [--output ket_qua.mp4]
```

### 3.4. Phím tắt khi chạy (Webcam/Video)
- `q` hoặc `ESC`: Thoát chương trình (đồng thời tự động vẽ và lưu Dashboard báo cáo tại `reports/emotion_report.png`).
- `s`: Chụp ảnh màn hình giao diện nhận diện (lưu tại thư mục `data/`).
- `SPACE` (Dấu cách): Tạm dừng / Tiếp tục chạy video (chỉ áp dụng ở `--mode video`).

---

## 4. Giải thuật cốt lõi

- **Haar Cascade Face Detector:** Nhận diện nhanh vị trí khuôn mặt trên CPU trong khoảng ~1-2ms.
- **CLAHE Preprocessing:** Cân bằng sáng cục bộ lưới 8x8, chống bóng đổ và ngược sáng webcam học sinh.
- **8-to-6 Remapping & Softmax:** Loại bỏ 2 nhãn thừa *Fear* (Sợ hãi) và *Contempt* (Khinh bỉ) từ đầu ra 8 lớp của HSEmotion, tái chuẩn hóa thang đo về đúng chuẩn phân phối xác suất 100% cho 6 lớp đích.
- **Temporal Smoothing:** Sử dụng hàng đợi deque trượt 10 khung hình tính xác suất trung bình, loại bỏ hiện tượng nháy nhãn hiển thị (Flickering).
- **Centroid Tracking:** Định danh khuôn mặt chính diện (Primary Face) dựa trên lưới tọa độ tâm 60px để lọc bỏ người đi lại phía sau hậu cảnh.
- **Adaptive Thresholding:** Ngưỡng quyết định riêng biệt cho từng lớp để chống báo động cảm xúc giả (`Happiness: 0.3, Sadness: 0.2, Surprise: 0.3, Anger: 0.3, Neutral: 0.35, Disgust: 0.3`).

---

## 5. Kết quả thực nghiệm định lượng

Hệ thống được đánh giá trên **Tập kiểm thử Đông Nam Á độc lập** gồm 150 ảnh (25 ảnh cho mỗi lớp cảm xúc) được trích chọn từ tập FairFace chủng tộc Southeast Asian, nhóm tuổi 10-29 và gán nhãn thủ công.

### 5.1. Báo cáo phân loại chi tiết (Classification Report)
- **Độ chính xác toàn cục (Accuracy):** **81.33%**
- **Macro F1-Score:** **0.81**

| Lớp cảm xúc | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Anger** | 0.92 | 0.96 | 0.94 | 25 |
| **Disgust** | 0.88 | 0.56 | 0.68 | 25 |
| **Happiness** | 1.00 | 1.00 | 1.00 | 25 |
| **Neutral** | 0.67 | 0.64 | 0.65 | 25 |
| **Sadness** | 0.64 | 1.00 | 0.78 | 25 |
| **Surprise** | 0.90 | 0.72 | 0.80 | 25 |

### 5.2. Nhận xét
- Việc tinh chỉnh bộ ngưỡng thích nghi thực nghiệm giúp tối ưu hóa sự cân bằng giữa Precision và Recall.
- Lớp **Happiness** và **Sadness** đều đạt độ nhạy (Recall) tuyệt đối 100% trên tập kiểm thử, đảm bảo không bỏ sót bất kỳ nét mặt tích cực hay biểu cảm buồn chán, ủ rũ nào của học sinh.
