# 📘 Hướng Dẫn Bảo Vệ Dự Án: Nhận Diện Cảm Xúc Người Học Online

**Môn học:** Khai thác thông tin đa phương tiện  
**Công nghệ:** Python + OpenCV + HSEmotion (EfficientNet-B0) + PyTorch  
**Phân chia công việc:** Bạn (40% - Dữ liệu, CLAHE, Biểu đồ & Báo cáo) | Quang (60% - Core AI, Softmax & Tối ưu hóa Webcam)

---

## 📌 Mục Lục

1. [Tổng Quan Hệ Thống](#1-tổng-quan-hệ-thống)
2. [Cấu Trúc Thư Mục Thực Tế](#2-cấu-trúc-thư-mục-thực-tế)
3. [Sơ Đồ Luồng Dữ Liệu (Data Flow)](#3-sơ-đồ-luồng-dữ-liệu-data-flow)
4. [Giải Thích Chi Tiết Từng File & Từng Hàm](#4-giải-thích-chi-tiết-từng-file--từng-hàm)
   - [4.1. `main.py` — File điều hướng chính (Entry Point)](#41-mainpy)
   - [4.2. `src/constants.py` — Định nghĩa hằng số](#42-srcconstantspy)
   - [4.3. `src/capture.py` — Xử lý Camera & OpenCV](#43-srccapturepy)
   - [4.4. `src/emotion_engine.py` — Bộ xử lý cảm xúc (Trọng tâm)](#44-srcemotion_enginepy)
   - [4.5. `src/file_tuner.py` — Huấn luyện tinh chỉnh (Fine-tuning)](#45-srcfile_tunerpy)
   - [4.6. `src/evaluator.py` — Kiểm thử & Đánh giá khoa học](#46-srcevaluatorpy)
   - [4.7. `src/emotion_stats.py` — Dashboard thống kê học tập](#47-srcemotion_statspy)
5. [Giải Thích Các Khái Niệm Chuyên Môn FER](#5-giải-thích-các-khái-niệm-chuyên-môn-fer)
6. [Bộ Câu Hỏi Bảo Vệ Hội Đồng & Câu Trả Lời Gợi Ý](#6-bộ-câu-hỏi-bảo-vệ-hội-đồng--câu-trả-lời-gợi-ý)

---

## 1. Tổng Quan Hệ Thống

### 1.1. Dự án này làm gì?
Đây là hệ thống **nhận diện cảm xúc người học thời gian thực** hỗ trợ lớp học trực tuyến (E-learning). Khi người học ngồi học trước Webcam:
1. **Phát hiện khuôn mặt:** Quét và xác định tọa độ khuôn mặt bằng thuật toán Haar Cascade.
2. **Tiền xử lý ảnh:** Cân bằng sáng thích nghi (CLAHE) để khử bóng đổ và nhiễu do đèn phòng học lệch.
3. **Nhận dạng cảm xúc:** Sử dụng mạng nơ-ron học sâu **EfficientNet-B0** để dự đoán xác suất của 6 biểu cảm mục tiêu: *Anger (Tức giận), Disgust (Ghê tởm), Happiness (Vui vẻ), Neutral (Bình thường), Sadness (Buồn bã), Surprise (Ngạc nhiên)*.
4. **Ổn định hiển thị:** Sử dụng các bộ lọc làm mịn thời gian (Temporal Smoothing) và định danh tâm khuôn mặt (Centroid Tracking) để hiển thị mượt mà trên webcam.
5. ** Dashboard báo cáo:** Khi buổi học kết thúc, xuất ra một Dashboard đồ họa thống kê diễn biến cảm xúc của học sinh để giáo viên đánh giá mức độ tương tác.

### 1.2. Công nghệ sử dụng
* `opencv-python` (4.9.0): Đọc camera, xử lý ảnh xám, bộ lọc màu, vẽ hộp bao (bounding boxes).
* `torch` & `torchvision` (PyTorch): Nạp mô hình, lan truyền xuôi (Inference), huấn luyện lan truyền ngược (Backpropagation).
* `hsemotion` (0.2.0): Thư viện chứa mô hình học sâu gốc EfficientNet-B0 đã được huấn luyện sẵn trên $450.000$ ảnh AffectNet.
* `scikit-learn`: Tính toán ma trận nhầm lẫn (Confusion Matrix) và bảng báo cáo phân loại (Classification Report).
* `matplotlib`: Vẽ biểu đồ Dashboard phân bố cảm xúc (Pie Chart) và biểu đồ đường thời gian (Timeline).

---

## 2. Cấu Trúc Thư Mục Thực Tế

Sau khi tinh giản các tính năng phụ để dự án đạt độ cô đọng cao nhất, cấu trúc thư mục gồm có:
```
nhandiencxuc/
├── main.py                     # File điều hướng chính (Entry Point)
├── requirements.txt            # Danh sách thư viện Python cần cài
├── dataset/                    # Tập dữ liệu người Đông Nam Á (lọc từ FairFace)
│   ├── train/                  # Tập huấn luyện & kiểm định (366 ảnh, 6 lớp)
│   └── test_dung1landuynhat/   # Tập kiểm thử độc lập (150 ảnh, 25 ảnh/lớp)
├── models/                     # Thư mục chứa trọng số mô hình sau Fine-tune
│   └── finetuned.pt            # File checkpoint trọng số (.pt)
├── reports/                    # Thư mục chứa báo cáo đánh giá & Dashboard
│   ├── confusion_matrix_*.png  # Biểu đồ ma trận nhầm lẫn
│   ├── training_curve.png      # Biểu đồ đường cong học tập
│   └── emotion_report.png      # Dashboard thống kê phiên học Webcam
├── src/                        # Thư mục chứa mã nguồn chính
│   ├── constants.py            # Hằng số màu sắc, nhãn hiển thị
│   ├── capture.py              # Xử lý luồng Camera OpenCV
│   ├── emotion_engine.py       # Bộ xử lý trung tâm (CLAHE, Smoothing, Tracking, Thresholds)
│   ├── file_tuner.py           # Logic huấn luyện Fine-tuning (Linear Probing)
│   ├── evaluator.py            # Logic chạy đánh giá & vẽ Confusion Matrix
│   └── emotion_stats.py        # Logic lưu trữ RAM & vẽ Dashboard Matplotlib
```

---

## 3. Sơ Đồ Luồng Dữ Liệu (Data Flow)

### 3.1. Luồng chạy Webcam thời gian thực
```
[Webcam (640x480)]
       │
       ▼
[Kiểm tra Frame Skip] ──► (Bỏ qua frame lẻ: Chỉ vẽ lại kết quả cũ để tăng FPS)
       │ (Hợp lệ - Frame chẵn)
       ▼
[Haar Cascade Face Detector]
       │
       ▼ (Lọc khuôn mặt kích thước >= 48x48)
[Tiền xử lý CLAHE] (BGR -> Xám -> Tăng tương phản ô 8x8 -> RGB)
       │
       ▼
[Mô hình EfficientNet-B0] ──► (Dự đoán raw xác suất)
       │
       ▼
[Temporal Smoothing & Centroid Tracking] (Lấy trung bình trượt 10 frame gần nhất)
       │
       ▼
[Adaptive Thresholding] ──► (Không đạt ngưỡng -> Đưa về mặc định Neutral)
       │ (Hợp lệ)
       ▼
[Vẽ Bounding Box & Ghi nhận stats] ──► [Lưu lịch sử RAM (SessionStats)]
```

### 3.2. Luồng Huấn luyện & Đánh giá
```
[Tập Train: 366 ảnh] ──► [Fine-tuning (Đóng băng backbone, chỉ học lớp Classifier)] ──► [models/finetuned.pt]
                                                                                              │
                                                                                              ▼
[Tập Test: 150 ảnh] ──► [Đánh giá chéo 2 mô hình (Original vs Fine-tuned)] ◄──────────────────┘
                               │
                               ▼
               [Reports: Confusion Matrix & Accuracy]
```

---

## 4. Giải Thích Chi Tiết Từng File & Từng Hàm

### 4.1. `main.py`
* **Hàm `main()` (Dòng 314):** Điểm khởi đầu chương trình. Gọi `build_parser()` để đọc tham số dòng lệnh và điều hướng chạy hàm xử lý tương ứng thông qua từ điển `dispatch` (`webcam` -> `run_webcam`, `finetune` -> `run_finetune`, `evaluate` -> `run_evaluate`).
* **Hàm `run_webcam(args)` (Dòng 37):** Vòng lặp Webcam chính. Thiết lập cơ chế **Frame Skipping (`skip_frames = 2`)** — chỉ chạy AI nhận diện trên các khung hình chẵn, khung hình lẻ chỉ vẽ đè tọa độ cũ nhằm kéo tốc độ xử lý lên mức mượt mà ~30 FPS trên CPU thường.
* **Hàm `_draw_webcam_overlay(...)` (Dòng 109):** Vẽ khung bounding box màu quanh mặt học sinh (màu sắc tự thay đổi theo cảm xúc lấy từ `EMOTION_COLORS`) và in nhãn kèm độ tin cậy. Vẽ thêm bảng thông số FPS, số khuôn mặt ở góc trên.
* **Hàm `_handle_key(...)` (Dòng 146):** Lọc phím bấm từ người dùng: `q`/`ESC` để thoát an toàn, `s` để chụp ảnh màn hình lưu vào thư mục `data/`, và `r` để giải phóng/reset bộ đệm làm mịn cảm xúc.

### 4.2. `src/constants.py`
* **`EMOTION_COLORS`:** Bảng màu định dạng BGR của OpenCV (ví dụ: `Happy` là màu xanh lá `(0, 255, 0)`, `Angry` là đỏ `(0, 0, 255)`).
* **`EMOTION_DISPLAY_MAP`:** Bản đồ ánh xạ tên cảm xúc để hiển thị giao diện thân thiện với người dùng (ví dụ: đổi `Happiness` thành `Happy`, `Sadness` thành `Sad`, `Anger` thành `Angry`).
* **`EMOTION_LABELS_6`:** Danh sách 6 cảm xúc mục tiêu chuẩn của hệ thống sau tinh chỉnh.

### 4.3. `src/capture.py`
* **Lớp `CameraCapture`:** Quản lý phần cứng camera.
  * `open_camera()` (Dòng 16): Mở camera thông qua backend `cv2.CAP_DSHOW` (DirectShow) để tăng độ ổn định trên hệ điều hành Windows, thiết lập độ phân giải chuẩn $640 \times 480$ pixel.
  * `read_frame()` (Dòng 37): Đọc khung hình mới dưới dạng ma trận điểm ảnh (Numpy Array).
  * `release_camera()` (Dòng 50): Giải phóng phần cứng webcam. Đây là hàm an toàn nằm trong khối `finally` của `main.py` để tránh lỗi khóa camera cho lần chạy tiếp theo.

### 4.4. `src/emotion_engine.py`
Đây là bộ não trung tâm điều khiển toàn bộ logic AI và xử lý hình ảnh của dự án.
* **Hàm `__init__(self, ...)` (Dòng 22):** Khởi tạo mô hình HSEmotion và nạp bộ trọng số tinh chỉnh từ `models/finetuned.pt` (nếu có). Nạp bộ lọc Haar Cascade tìm mặt và khai báo tham số bộ lọc tương phản sáng CLAHE.
* **Hàm `_preprocess_face(self, face_img)` (Dòng 109):** **Công việc của Bạn.** Chuyển vùng khuôn mặt cắt ra thành ảnh xám $\rightarrow$ Gọi bộ lọc CLAHE (`self.clahe.apply(gray)`) để cân bằng sáng cục bộ trong lưới $8 \times 8$ ô nhỏ với giới hạn tương phản `2.0` $\rightarrow$ Chuyển ngược lại RGB để làm dữ liệu đầu vào cho mạng EfficientNet-B0.
* **Hàm `_get_face_key(self, x, y, w, h)` (Dòng 119):** **Công việc của Quang.** Thuật toán theo dõi khuôn mặt bằng tâm hình học (Centroid Tracking). Tính tọa độ tâm khuôn mặt, lượng tử hóa theo lưới kích thước `GRID_SIZE = 60` pixel. Điều này giúp hệ thống xác định cùng 1 người kể cả khi khuôn mặt bị rung lắc nhẹ trước camera.
* **Hàm `predict_emotion(self, face_img, face_key)` (Dòng 125):** Tiền xử lý ảnh $\rightarrow$ Chạy mô hình để lấy xác suất cảm xúc. Đưa xác suất vào hàng đợi trượt `deque` (`BUFFER_SIZE = 10`) để lấy trung bình cộng xác suất của 10 khung hình gần nhất (**Temporal Smoothing**). Cuối cùng đối chiếu điểm số với bảng **ngưỡng quyết định thích nghi (Adaptive Thresholding)**; nếu điểm không vượt qua ngưỡng riêng của lớp cảm xúc đó, hệ thống sẽ tự động đưa kết quả về nhãn mặc định an toàn là `Neutral`.

### 4.5. `src/file_tuner.py`
Quản lý logic huấn luyện Fine-tuning mô hình AI.
* **Hàm `_load_model(model_name)` (Dòng 76):** Hàm tĩnh để tạo kiến trúc mạng EfficientNet-B0 thông qua thư viện `timm`. Sửa lỗi lệch pha tên khóa nơ-ron lớp Classifier giữa các phiên bản thư viện cũ và mới.
* **Hàm `load_dataset(...)` (Dòng 106):** Nạp dữ liệu từ thư mục `dataset/train`. Áp dụng các phép biến đổi ảnh (Resize 224x224, chuẩn hóa ImageNet). Remap nhãn từ cấu trúc thư mục tương thích với 6 lớp mục tiêu. Tự động chia tập dữ liệu thành 85% Train ($312$ ảnh) và 15% Val ($54$ ảnh).
* **Hàm `train(...)` (Dòng 175):** Vòng lặp huấn luyện lan truyền ngược (Backpropagation). Áp dụng kỹ thuật **Linear Probing** bằng cách đóng băng toàn bộ backbone, chỉ mở duy nhất lớp Classifier cuối để học. So sánh val accuracy sau mỗi epoch, chỉ lưu file `finetuned.pt` khi đạt độ chính xác cao nhất (Best Checkpoint).

### 4.6. `src/evaluator.py`
Hỗ trợ kiểm thử và đánh giá hiệu năng khoa học của mô hình.
* **Hàm `evaluate_and_compare(...)` (Dòng 44):** Chạy kiểm thử độc lập cả 2 mô hình (Original và Fine-tuned) trên tập test độc lập `test_dung1landuynhat` (150 ảnh). In bảng Classification Report (Precision, Recall, F1-Score) và gọi hàm vẽ đồ thị.
* **Hàm `_plot_confusion_matrix(...)` (Dòng 89):** Vẽ biểu đồ ma trận nhầm lẫn (Confusion Matrix) dạng heatmap sử dụng Seaborn, thể hiện các lớp dự đoán đúng trên đường chéo chính và lỗi dự đoán sai lệch ngoài đường chéo.

### 4.7. `src/emotion_stats.py`
* **Hàm `record(...)` (Dòng 45):** **Công việc của Bạn.** Ghi nhận nhãn cảm xúc và độ tin cậy của khuôn mặt chính diện (Primary Face) vào bộ nhớ đệm RAM (`self.history`) với độ phức tạp $O(1)$ (phép `append` danh sách), đảm bảo ghi nhận liên tục mà không gây nghẽn cổ chai ghi đĩa cứng, giúp giữ nguyên FPS webcam.
* **Hàm `generate_report(...)` (Dòng 63):** **Công việc của Bạn.** Khi kết thúc buổi học, Matplotlib sẽ xử lý lịch sử lưu trong RAM để vẽ và xuất ra Dashboard đồ thị `reports/emotion_report.png` gồm:
  1. Biểu đồ tròn (Pie Chart) thể hiện phân bố phần trăm thời gian của các trạng thái cảm xúc.
  2. Biểu đồ dòng thời gian (Timeline) kết hợp đường chạy trạng thái (Step Line) và các điểm chấm tròn phân tán (Scatter Plot) với kích thước chấm co dãn động theo độ tin cậy (confidence) của AI.

---

## 5. Giải Thế Các Khái Niệm Chuyên Môn FER

* **Haar Cascade:** Thuật toán phát hiện khuôn mặt cổ điển dựa trên sự chênh lệch độ sáng tối của các vùng hình học trên mặt (như vùng mắt thường tối hơn vùng trán). Chạy rất nhẹ trên CPU thường nhưng kém chính xác khi quay nghiêng góc quá $30^\circ$.
* **CLAHE:** Phương pháp tăng độ tương phản hình ảnh cục bộ. Thay vì cân bằng sáng toàn bộ ảnh làm tăng nhiễu hạt, CLAHE chia khuôn mặt thành lưới $8 \times 8$ ô nhỏ, giới hạn độ tương phản tối đa tại ngưỡng `2.0` để triệt tiêu nhiễu hạt (muỗi camera) và áp dụng nội suy song tuyến tính để làm mịn các ranh giới ô.
* **Linear Probing:** Kỹ thuật chuyển giao tri thức (Transfer Learning). Đóng băng toàn bộ trọng số đặc trưng cơ mặt đã học từ $450.000$ ảnh AffectNet của phần Backbone, chỉ mở duy nhất lớp Classifier (gồm $7.686$ tham số) để học cách ánh xạ sang 6 lớp cảm xúc của bộ dữ liệu mới. Tránh quá khớp (Overfitting) khi tập dữ liệu huấn luyện nhỏ ($366$ ảnh).
* **Subspace Projection (Phép chiếu không gian con):** Vì mô hình gốc đầu ra có 8 lớp cảm xúc (chứa cả *Fear* và *Contempt*) trong khi mô hình Fine-tuned và tập test chỉ cần đánh giá 6 lớp. Để so sánh sòng phẳng và công bằng toán học, hệ thống loại bỏ 2 cổng điểm số của lớp thừa, trích xuất logits của 6 lớp cảm xúc mục tiêu và tính toán lại hàm Softmax để tổng xác suất của 6 lớp này quy về lại đúng $100\%$ trước khi so sánh.
* **Overfitting (Quá khớp / Học vẹt):** Hiện tượng mô hình đạt độ chính xác gần như tuyệt đối trên tập huấn luyện (Train) nhưng đoán sai lệch khi gặp dữ liệu thực tế mới (Test). Khắc phục bằng cách đóng băng backbone, thiết lập learning rate nhỏ ($10^{-4}$), và huấn luyện số epoch vừa đủ (15 epochs).

---

## 6. Bộ Câu Hỏi Bảo Vệ Hội Đồng & Câu Trả Lời Gợi Ý

### ❓ Câu 1: Tại sao nhóm em chỉ huấn luyện mô hình có 15 Epochs mà không phải là 100 Epochs để đạt độ chính xác cao hơn?
**💡 Trả lời gợi ý:**  
*Dạ thưa thầy/cô, vì dự án của nhóm em áp dụng chiến lược đóng băng toàn bộ backbone (Linear Probing), hệ thống chỉ cần cập nhật duy nhất lớp phân loại tuyến tính cuối cùng với số lượng tham số rất nhỏ ($7.686$ tham số). Lớp phân loại này hội tụ cực kỳ nhanh, thực nghiệm cho thấy sau Epoch thứ 10, hàm mất mát (Loss) đã đi ngang và đạt mức tối ưu. Do đó, việc dừng ở Epoch 15 là lựa chọn tối ưu nhằm bảo toàn khả năng tổng quát hóa của mô hình và ngăn ngừa hiện tượng học vẹt (overfitting) trên tập dữ liệu nhỏ 366 ảnh.*

### ❓ Câu 2: Chỉ số F1-Score của lớp Anger (Tức giận) lại bị sụt giảm nhẹ từ 0.96 xuống 0.90 sau khi tinh chỉnh. Nhóm em giải thích hiện tượng này như thế nào?
**💡 Trả lời gợi ý:**  
*Dạ thưa thầy/cô, mô hình gốc ban đầu vốn dĩ đã nhận diện cảm xúc Anger cực kỳ xuất sắc với điểm F1-Score rất cao là $0.96$. Khi tiến hành Fine-tune trên tập dữ liệu nhỏ tự thu thập, thuật toán tối ưu hóa (Adam) phải giải quyết bài toán tối ưu hóa hàm mất mát toàn cục (Global Loss Function) cho cả 6 lớp cảm xúc để kéo tổng hiệu năng trung bình của hệ thống lên. Do đó, mô hình đã điều chỉnh lại các ranh giới phân loại, chấp nhận "san sẻ" độ tập trung của lớp đã quá tốt (Anger giảm từ 0.96 xuống 0.90) để cải thiện đáng kể các lớp khó khác như Disgust (F1-score tăng vượt trội từ 0.63 lên 0.76) và Sadness (tăng từ 0.74 lên 0.78). Đây là hiện tượng đánh đổi (trade-off) rất bình thường và hợp lý trong học máy để đạt được hiệu năng tổng thể tốt nhất.*

### ❓ Câu 3: Làm thế nào nhóm em so sánh được điểm số giữa Mô hình gốc (8 cổng đầu ra) và Mô hình tinh chỉnh (6 cổng đầu ra) một cách công bằng?
**💡 Trả lời gợi ý:**  
*Dạ thưa thầy/cô, nếu so sánh trực tiếp điểm số xác suất thô sẽ mất công bằng toán học vì tổng xác suất phân phối của mô hình gốc bị phân tán sang cả 2 lớp thừa là Fear và Contempt. Nhóm em đã thiết lập thuật toán Phép chiếu không gian con và Tái chuẩn hóa (Subspace Projection & Re-normalisation). Hệ thống loại bỏ điểm số của 2 cổng thừa này, chỉ lấy logits của 6 cảm xúc mục tiêu giống mô hình tinh chỉnh, rồi chia lại tỷ lệ để tổng xác suất của 6 lớp cảm xúc này quy về lại đúng $100\%$. Nhờ vậy, hai mô hình được so sánh trên cùng một hệ quy chiếu và cùng một tập kiểm thử độc lập 150 ảnh.*

### ❓ Câu 4: Bộ lọc làm mịn thời gian (Temporal Smoothing) có ưu điểm gì? Nhược điểm của nó là gì và nhóm đã khắc phục thế nào?
**💡 Trả lời gợi ý:**  
*Dạ thưa thầy/cô, bộ lọc làm mịn thời gian giúp triệt tiêu hiện tượng nháy nhãn (Label Flickering) gây ra bởi các nhiễu động ánh sáng tức thời hoặc cử động chớp mắt của học sinh, giúp hiển thị cảm xúc mượt mà. Tuy nhiên, nhược điểm của nó là gây ra độ trễ phản hồi (Time Lag) khi học sinh chuyển đột ngột từ cảm xúc này sang cảm xúc khác (vì bộ đệm cần thời gian tích lũy khung hình mới). Nhóm em đã cấu hình kích thước bộ đệm tối ưu là `10` khung hình (tương đương khoảng $0.3 - 0.6$ giây ở tốc độ webcam thực tế), đây là khoảng thời gian đủ ngắn để mắt người không cảm nhận được độ trễ, nhưng đủ dài để lọc nhiễu hiệu quả.*

### ❓ Câu 5: Hệ thống của nhóm em hoạt động ra sao nếu trong khung hình Webcam xuất hiện nhiều người (ví dụ có người đi lại phía sau học sinh)?
**💡 Trả lời gợi ý:**  
*Dạ thưa thầy/cô, nhóm em đã cài đặt thuật toán theo dõi tâm khuôn mặt (Centroid Tracking) kết hợp cơ chế lọc khuôn mặt đại diện (Primary Face). Hệ thống tính toán tâm hình học của các hộp bao khuôn mặt và theo dõi độc lập từng người trên lưới `60` pixel. Trong đó, hệ thống quy ước khuôn mặt đầu tiên phát hiện được (chính diện camera) là Primary Face đại diện cho học sinh học bài. Chỉ có dữ liệu cảm xúc của Primary Face mới được ghi nhận vào RAM phục vụ vẽ Dashboard đồ thị cuối phiên học. Mọi khuôn mặt phụ xuất hiện ở hậu cảnh sẽ tự động bị bộ lọc loại bỏ khỏi biểu đồ, đảm bảo tính tập trung và chính xác của dữ liệu báo cáo.*
