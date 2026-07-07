# -*- coding: utf-8 -*-
"""
main_note.py - Phiên bản ghi chú chi tiết từng dòng của Hệ thống nhận dạng cảm xúc người học online
Sử dụng: python main_note.py [--mode webcam|image|video]
"""

import os, sys, time, argparse, numpy as np, cv2, torch  # [Dòng 6] Nạp thư viện: os (quản lý file/thư mục), sys (biến hệ thống), time (đo thời gian), argparse (đọc tham số dòng lệnh), numpy (ma trận toán học), cv2 (OpenCV), torch (học sâu PyTorch)
from collections import deque, Counter  # [Dòng 7] deque: hàng đợi trượt 10 khung hình để làm mịn nhãn hiển thị, Counter: bộ đếm số lần xuất hiện cảm xúc phục vụ vẽ biểu đồ tròn
from datetime import datetime  # [Dòng 8] Lấy ngày giờ hệ thống thực tế để tự động đặt tên file ảnh chụp màn hình khi người dùng nhấn phím 's'
from hsemotion.facial_emotions import HSEmotionRecognizer  # [Dòng 9] Nạp lớp HSEmotionRecognizer chứa mô hình EfficientNet-B0 pretrained để dự đoán cảm xúc
import matplotlib.pyplot as plt  # [Dòng 10] Nạp thư viện vẽ đồ thị Matplotlib để tự động xuất ra Dashboard báo cáo PNG sau khi học xong

# ── Constants ──  # [Dòng 12] Khai báo các hằng số cấu hình toàn cục của hệ thống
EMOTIONS = ['Anger','Disgust','Happiness','Neutral','Sadness','Surprise']  # [Dòng 13] Danh sách 6 nhãn cảm xúc đích mà hệ thống sẽ phân loại và vẽ lên màn hình
EMO_COLORS = {'Happy':(0,255,0),'Sad':(255,80,80),'Angry':(0,0,255),  # [Dòng 14] Bản đồ màu BGR vẽ bounding box: Happy (Xanh lá), Sad (Xanh dương nhạt), Angry (Đỏ)
              'Surprise':(0,165,255),'Neutral':(180,180,180),'Disgust':(0,160,160)}  # [Dòng 15] Tiếp tục mã màu BGR: Surprise (Cam), Neutral (Xám), Disgust (Vàng đất)
DISPLAY = {'Anger':'Angry','Happiness':'Happy','Sadness':'Sad','Contempt':'Disgust'}  # [Dòng 16] Ánh xạ đổi tên nhãn nội bộ viết tắt sang tên hiển thị thân thiện trên giao diện camera
FACE_CASCADE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # [Dòng 17] Đường dẫn tệp XML Haar Cascade dùng để quét và tìm vị trí khuôn mặt trong OpenCV

# ── Emotion Engine ──  # [Dòng 19] Khai báo Class EmotionEngine - bộ xử lý kỹ thuật, tiền xử lý và chạy mô hình dự đoán
class EmotionEngine:  # [Dòng 20] Định nghĩa lớp EmotionEngine
    def __init__(self, model_name='enet_b0_8_best_afew'):  # [Dòng 21] Hàm khởi tạo các thành phần của engine khi đối tượng được tạo
        # Override torch.load to bypass weights_only warning in HSEmotionRecognizer  # [Dòng 22] Ghi chú kỹ thuật tắt cảnh báo PyTorch
        torch_load = torch.load  # [Dòng 23] Lưu tạm hàm torch.load gốc của thư viện PyTorch vào biến
        torch.load = lambda f,**kw: torch_load(f,**{**kw,'weights_only':False})  # [Dòng 24] Ghi đè hàm load bằng lambda để ép weights_only=False, tắt cảnh báo bảo mật
        self.er = HSEmotionRecognizer(model_name=model_name)  # [Dòng 25] Khởi tạo mô hình HSEmotion (EfficientNet-B0 pretrained ~25MB)
        torch.load = torch_load  # [Dòng 26] Khôi phục lại hàm torch.load gốc của PyTorch ngay lập tức để không ảnh hưởng hàm khác
        
        self.fc = cv2.CascadeClassifier(FACE_CASCADE)  # [Dòng 28] Nạp bộ quét mặt Haar Cascade từ file XML đã định nghĩa ở dòng 17
        self.bufs = {}  # [Dòng 29] Dict bufs: từ điển lưu trữ hàng đợi trượt deque lọc mịn xác suất cho từng khuôn mặt
        self.BSZ, self.GRID = 10, 60  # [Dòng 30] BSZ = 10 (cỡ hàng đợi làm mịn), GRID = 60 (kích thước ô lưới ảo để tracking định danh mặt)
        self.THRESH = {'Happiness':.3,'Sadness':.2,'Surprise':.3,'Anger':.3,'Neutral':.35,'Disgust':.3}  # [Dòng 31] Bộ ngưỡng quyết định thích nghi cho 6 cảm xúc để tránh báo giả
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # [Dòng 32] Khởi tạo CLAHE cân bằng sáng cục bộ: giới hạn tương phản = 2.0, lưới 8x8 ô

    def detect(self, frame):  # [Dòng 34] Định nghĩa hàm phát hiện các khuôn mặt có trong một khung hình cấp vào
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # [Dòng 35] Chuyển ảnh màu BGR sang ảnh xám vì Haar Cascade chỉ cần dùng cường độ sáng để quét
        return self.fc.detectMultiScale(gray, 1.1, 5, minSize=(30,30))  # [Dòng 36] Quét tìm mặt: scaleFactor=1.1, minNeighbors=5, bỏ qua mặt < 30x30px

    def predict(self, face_img):  # [Dòng 38] Định nghĩa hàm dự đoán cảm xúc cho một vùng ảnh khuôn mặt đã cắt ra
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)  # [Dòng 39] Chuyển ảnh cắt khuôn mặt sang ảnh xám Grayscale để chuẩn bị cân bằng sáng
        eq = self.clahe.apply(gray)  # [Dòng 40] Áp dụng CLAHE để cân bằng sáng đều trên các vùng mặt, làm rõ nét các nếp nhăn và thớ cơ biểu cảm
        rgb = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)  # [Dòng 41] Chuyển lại ảnh xám sang 3 kênh RGB vì model EfficientNet yêu cầu đầu vào 3 kênh màu
        name, scores = self.er.predict_emotions(rgb, logits=False)  # [Dòng 42] Chạy model predict trả về mảng 8 xác suất cảm xúc gốc
        # Anh xa 8 lop cua HSEmotion sang 6 lop cua bai toan  # [Dòng 43] Ghi chú phép ánh xạ
        raw6 = np.array([scores[0], scores[2], scores[4], scores[5], scores[6], scores[7]])  # [Dòng 44] Lọc lấy 6 lớp, bỏ Fear (chỉ số 1) và Contempt (chỉ số 3)
        scores = raw6 / raw6.sum() if raw6.sum() > 0 else raw6  # [Dòng 45] Tái chuẩn hóa Softmax rút gọn: chia từng phần tử cho tổng để tổng xác suất = 100%
        return scores  # [Dòng 46] Trả về mảng 6 giá trị xác suất đã chuẩn hóa của khuôn mặt

    def process(self, frame):  # [Dòng 48] Định nghĩa hàm tổng xử lý luồng trên 1 khung hình (detect, tracking, predict, smoothing, threshold)
        results = []  # [Dòng 49] Khởi tạo danh sách kết quả trống cho khung hình hiện tại
        faces = self.detect(frame)  # [Dòng 50] Gọi hàm detect ở dòng 34 để phát hiện tất cả các khuôn mặt trong frame
        if len(faces) > 1:  # [Dòng 51] Nếu phát hiện từ 2 khuôn mặt trở lên
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)  # [Dòng 52] Sắp xếp mặt theo diện tích giảm dần (f[2]*f[3]). Mặt lớn nhất ở vị trí đầu (index 0)
        active = set()  # [Dòng 53] Khởi tạo tập hợp active để lưu trữ tọa độ tâm của các khuôn mặt đang thực sự xuất hiện trên frame
        for (x,y,w,h) in faces:  # [Dòng 54] Vòng lặp xét qua từng khuôn mặt
            if w>=48 and h>=48:  # [Dòng 55] Chỉ quan tâm đến các khuôn mặt có độ rộng và dài từ 48px trở lên
                k = ((x+w//2)//self.GRID*self.GRID, (y+h//2)//self.GRID*self.GRID)  # [Dòng 56] Centroid Tracking: Lượng tử hóa tâm mặt về ô lưới 60px làm khóa định danh k
                active.add(k)  # [Dòng 57] Thêm khóa định danh k vào tập hợp active
        for k in set(self.bufs.keys())-active:  # [Dòng 58] Tìm các khuôn mặt có trong bộ đệm từ điển nhưng không còn xuất hiện trên camera
            del self.bufs[k]  # [Dòng 59] Xóa bộ đệm của khuôn mặt đã rời đi khỏi camera để tránh bị rò rỉ bộ nhớ RAM
        for (x,y,w,h) in faces:  # [Dòng 60] Vòng lặp thứ hai để thực hiện predict cho từng mặt
            if w<48 or h<48: continue  # [Dòng 61] Bỏ qua ngay các khuôn mặt ở quá xa, quá nhỏ (nhỏ hơn 48x48 pixel)
            k = ((x+w//2)//self.GRID*self.GRID, (y+h//2)//self.GRID*self.GRID)  # [Dòng 62] Tính khóa định danh k để định vị đúng người trong bufs
            face = frame[y:y+h,x:x+w]  # [Dòng 63] Cắt lấy riêng vùng ảnh ma trận chứa khuôn mặt (Crop) từ frame ảnh gốc
            scores = self.predict(face)  # [Dòng 64] Chạy hàm predict ở dòng 38 để lấy mảng 6 xác suất cảm xúc
            if k not in self.bufs: self.bufs[k] = deque(maxlen=self.BSZ)  # [Dòng 65] Nếu là người mới, khởi tạo hàng đợi deque lưu tối đa 10 khung hình
            self.bufs[k].append(scores)  # [Dòng 66] Đẩy mảng xác suất mới dự đoán vào hàng đợi deque của người đó
            avg = np.mean(self.bufs[k], axis=0)  # [Dòng 67] Temporal Smoothing: Tính trung bình cộng xác suất của 10 khung hình gần nhất
            idx = np.argmax(avg)  # [Dòng 68] Tìm chỉ số index của cảm xúc đạt xác suất trung bình lớn nhất
            conf = float(avg[idx])  # [Dòng 69] Lấy giá trị xác suất cao nhất đó (độ tin cậy) ép kiểu Float
            emo = EMOTIONS[idx]  # [Dòng 70] Tra cứu tên cảm xúc từ mảng EMOTIONS dựa vào chỉ số idx vừa tìm được
            if conf < self.THRESH.get(emo, .4):  # [Dòng 71] Kiểm tra ngưỡng thích nghi: nếu độ tin cậy thấp hơn ngưỡng quy định (ví dụ Happiness < 0.3)
                emo, conf = 'Neutral', float(avg[EMOTIONS.index('Neutral')])  # [Dòng 72] Ép cảm xúc về nhãn mặc định Neutral và lấy đúng xác suất lớp Neutral tương ứng
            else:  # [Dòng 73] Trường hợp đạt trên ngưỡng thích nghi
                emo = DISPLAY.get(emo, emo)  # [Dòng 74] Đổi tên sang nhãn hiển thị thân thiện trên UI (ví dụ: Happiness -> Happy)
            results.append({'bbox':(x,y,w,h),'emotion':emo,'confidence':conf,'probs':avg})  # [Dòng 75] Thêm kết quả gồm bounding box, nhãn, độ tin cậy vào list results
        return results  # [Dòng 76] Trả về danh sách kết quả phân tích các khuôn mặt của khung hình này

# ── Session Stats ──  # [Dòng 78] Khai báo Class SessionStats - ghi nhận thống kê phiên học và tự động vẽ đồ thị Dashboard
class SessionStats:  # [Dòng 79] Định nghĩa lớp SessionStats
    def __init__(self):  # [Dòng 80] Hàm khởi tạo bộ thống kê
        self.history = []  # [Dòng 81] Khởi tạo danh sách history trống để ghi nhận lịch sử cảm xúc theo thời gian thực trên RAM
        self.order = list(EMO_COLORS.keys())  # [Dòng 82] Lấy danh sách tên cảm xúc theo mã màu để sắp xếp thứ tự trục Y trên biểu đồ timeline

    def record(self, t, results):  # [Dòng 84] Định nghĩa hàm ghi nhận dữ liệu cảm xúc tại mốc thời gian t (giây)
        if results:  # [Dòng 85] Nếu có kết quả phát hiện mặt trên frame
            r = results[0]  # [Dòng 86] Chỉ lấy khuôn mặt lớn nhất ở vị trí results[0] (Primary Face - học sinh chính), bỏ qua nhiễu sau lưng
            self.history.append({'time':t,'emotion':r['emotion'],'confidence':r['confidence']})  # [Dòng 87] Append O(1) thời gian, nhãn, độ tin cậy vào RAM

    def report(self, out_dir='reports'):  # [Dòng 89] Định nghĩa hàm vẽ và lưu Dashboard reports/emotion_report.png
        if not self.history: return  # [Dòng 90] Nếu danh sách lịch sử trống (chưa ghi được gì), dừng hàm không vẽ
        os.makedirs(out_dir, exist_ok=True)  # [Dòng 91] Tạo thư mục reports/ trên ổ cứng nếu thư mục này chưa tồn tại
        ts = [h['time'] for h in self.history]  # [Dòng 92] Tách riêng danh sách thời gian ts từ lịch sử
        emos = [h['emotion'] for h in self.history]  # [Dòng 93] Tách riêng danh sách các nhãn cảm xúc emos từ lịch sử
        confs = [h['confidence'] for h in self.history]  # [Dòng 94] Tách riêng danh sách các độ tin cậy confs từ lịch sử
        cnt = Counter(emos)  # [Dòng 95] Đếm tần suất xuất hiện từng cảm xúc để vẽ biểu đồ tròn
        present = [e for e in self.order if e in cnt]  # [Dòng 96] Lọc các cảm xúc thực tế có xuất hiện trong phiên học
        counts = [cnt[e] for e in present]  # [Dòng 97] Lấy số lần xuất hiện tương ứng với từng cảm xúc đó
        colors = [tuple(c/255 for c in reversed(EMO_COLORS[e])) for e in present]  # [Dòng 98] Chuyển đổi mã màu BGR sang tỉ lệ RGB (0.0 - 1.0) của Matplotlib
        emo2y = {e:i for i,e in enumerate(self.order)}  # [Dòng 99] Ánh xạ tên cảm xúc sang index số nguyên 0-5 để làm tọa độ trục đứng Y
        yv = [emo2y[e] for e in emos]  # [Dòng 100] Tạo mảng yv chứa index trục đứng tương ứng cho toàn bộ chuôi cảm xúc lịch sử

        fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12,10),gridspec_kw={'height_ratios':[1,1.5]})  # [Dòng 102] Tạo khung biểu đồ size 12x10 inch, chia làm 2 đồ thị chồng dọc: ax1 (trên, tỉ lệ 1.0), ax2 (dưới, tỉ lệ 1.5)
        ax1.pie(counts, labels=present, colors=colors, autopct=lambda p:f'{p:.1f}%', startangle=90)  # [Dòng 103] Vẽ đồ thị tròn (Pie Chart) phân bố thời lượng cảm xúc. Góc bắt đầu 90 độ
        ax1.set_title('Phan Bo Cam Xuc')  # [Dòng 104] Đặt tên tiêu đề cho biểu đồ tròn là "Phan Bo Cam Xuc"
        ax2.step(ts, yv, where='post', color='gray', alpha=.5, linewidth=1.5, linestyle='--')  # [Dòng 105] Vẽ biểu đồ bước đường đứt nét xám nối các mốc cảm xúc để thấy sự biến đổi
        for t,e,c in zip(ts,emos,confs):  # [Dòng 106] Duyệt qua thời gian, nhãn, độ tin cậy để vẽ các chấm tròn
            rgb = tuple(v/255 for v in reversed(EMO_COLORS[e]))  # [Dòng 107] Lấy mã màu RGB phù hợp cho chấm tròn tương ứng với cảm xúc đó
            ax2.scatter(t, emo2y[e], s=max(40,c*250), color=[rgb], edgecolors='black', linewidth=.5)  # [Dòng 108] Vẽ chấm tròn, size s co giãn động theo độ tin cậy = max(40, c*250), có viền đen
        ax2.set_yticks(range(len(self.order)))  # [Dòng 109] Thiết lập số vạch chia trên trục Y bằng đúng số cảm xúc (0 đến 5)
        ax2.set_yticklabels(self.order)  # [Dòng 110] Gán chữ tên cảm xúc tương ứng vào các vạch chia trục Y
        ax2.set_xlabel('Thoi gian (giay)')  # [Dòng 111] Đặt tên nhãn trục nằm ngang X là "Thời gian (giây)"
        ax2.set_ylabel('Cam xuc')  # [Dòng 112] Đặt tên nhãn trục đứng Y là "Cảm xúc"
        ax2.grid(True, alpha=.3)  # [Dòng 113] Bật lưới biểu đồ với độ trong suốt 0.3 để dễ gióng hàng tọa độ
        plt.tight_layout()  # [Dòng 114] Căn chỉnh tự động để các chữ trên biểu đồ không bị chồng chéo hoặc mất viền
        plt.savefig(os.path.join(out_dir,'emotion_report.png'), dpi=150, bbox_inches='tight')  # [Dòng 115] Lưu toàn bộ hình ảnh Dashboard thành file reports/emotion_report.png với DPI = 150
        plt.close()  # [Dòng 116] Đóng đối tượng đồ thị Matplotlib để giải phóng hoàn toàn bộ nhớ RAM
        print(f'  Da luu: reports/emotion_report.png')  # [Dòng 117] In thông báo đã ghi file Dashboard ra màn hình console

# ── Draw Helpers ──  # [Dòng 119] Định nghĩa các hàm trợ giúp đồ họa vẽ khung bounding box và viết chữ nhãn lên giao diện camera
def draw_boxes(frame, results):  # [Dòng 120] Định nghĩa hàm draw_boxes, nhận vào khung hình và kết quả results
    out = frame.copy()  # [Dòng 121] Copy ra bản sao ảnh out để vẽ, giữ nguyên ma trận ảnh thô frame ban đầu khỏi bị thay đổi
    for r in results:  # [Dòng 122] Vòng lặp duyệt qua kết quả từng mặt hiện có trên khung hình
        x,y,w,h = r['bbox']; emo = r['emotion']; conf = r['confidence']  # [Dòng 123] Lấy tọa độ khung hộp (x,y,w,h), tên cảm xúc, và độ tin cậy
        color = EMO_COLORS.get(emo,(255,255,255))  # [Dòng 124] Lấy màu sắc vẽ khung tương ứng với cảm xúc, mặc định là trắng nếu không có
        cv2.rectangle(out,(x,y),(x+w,y+h),color,2)  # [Dòng 125] Vẽ khung hộp chữ nhật nét dày 2px bao quanh khuôn mặt học sinh
        lbl = f'{emo}: {conf:.0%}'  # [Dòng 126] Tạo chuỗi văn bản hiển thị: cảm xúc + độ tin cậy làm tròn phần trăm (ví dụ: Happy: 95%)
        (lw,lh),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,.65,2)  # [Dòng 127] Tính kích thước ô chữ để thiết kế nền cho chữ vừa vặn
        cv2.rectangle(out,(x,y-lh-10),(x+lw+6,y),color,-1)  # [Dòng 128] Vẽ ô hộp chữ nhật đặc (độ dày -1) làm nền cho chữ nhãn ngay trên khung mặt
        cv2.putText(out,lbl,(x+3,y-5),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),2)  # [Dòng 129] Viết nhãn chữ màu đen (0,0,0) lên trên ô hộp nền nét dày 2px
    return out  # [Dòng 130] Trả về khung hình đã được vẽ bounding box và ghi chú hoàn thiện

# ── Modes ──  # [Dòng 132] Khai báo các hàm điều khiển ứng dụng theo từng chế độ chạy đầu vào
def mode_webcam(args):  # [Dòng 133] Định nghĩa hàm chạy Webcam thời gian thực
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # [Dòng 134] Mở webcam bằng DirectShow (Windows) để camera khởi động lên ngay lập tức
    if not cap.isOpened(): cap = cv2.VideoCapture(args.camera)  # [Dòng 135] Thử lại bằng hàm mặc định nếu DirectShow không tương thích thiết bị
    if not cap.isOpened(): print('Ko mo duoc camera!'); return  # [Dòng 136] Nếu camera bị lỗi hoặc không tìm thấy, in lỗi và dừng chương trình
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)  # [Dòng 137] Cấu hình webcam chạy ở độ phân giải chuẩn 640x480 pixel
    eng = EmotionEngine(args.model)  # [Dòng 138] Khởi tạo bộ xử lý cảm xúc engine chạy model EfficientNet
    stats = SessionStats(); fc=0; skip=2; last=[]; start_t = time.time()  # [Dòng 139] Khởi tạo stats ghi chép, bộ đếm frame fc=0, skip=2 để nhảy frame tiết kiệm CPU, start_t ghi mốc bắt đầu

    os.makedirs('data', exist_ok=True)  # [Dòng 141] Tự động tạo thư mục data/ dùng để lưu các file ảnh chụp màn hình camera

    try:  # [Dòng 143] Khối block try an toàn bắt buộc chạy cấp camera
        while True:  # [Dòng 144] Vòng lặp vô hạn đọc webcam thời gian thực
            ret,frame = cap.read()  # [Dòng 145] Đọc một khung hình mới từ camera. ret báo đọc thành công, frame chứa ảnh
            if not ret: break  # [Dòng 146] Nếu webcam bị mất kết nối hoặc không đọc được ảnh, tự động thoát vòng lặp
            fc+=1  # [Dòng 147] Tăng biến đếm khung hình thêm 1 đơn vị
            if fc%skip==0: last=eng.process(frame); stats.record(time.time()-start_t,last)  # [Dòng 148] Nhảy khung hình: chỉ xử lý AI ở frame chẵn (2,4,6...) để đỡ nóng máy, ghi nhận stats
            display = draw_boxes(frame, last)  # [Dòng 149] Vẽ bounding box và nhãn lên ảnh thực tế để hiển thị
            cv2.putText(display,f'Frame:{fc} Faces:{len(last)}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)  # [Dòng 150] In thông số số frame và số mặt ở góc trên bên trái
            cv2.putText(display,'q:thoat s:chup',(10,display.shape[0]-12),cv2.FONT_HERSHEY_SIMPLEX,.45,(200,200,200),1)  # [Dòng 151] In chữ hướng dẫn phím tắt dưới đáy bên trái
            cv2.imshow('Emotion Detection',display)  # [Dòng 152] Mở cửa sổ giao diện camera hiển thị kết quả nhận diện cảm xúc cho người học xem
            k=cv2.waitKey(1)&0xFF  # [Dòng 153] Đọc phím bấm của người dùng thời gian chờ là 1 mili giây
            if k in (ord('q'),27): break  # [Dòng 154] Nếu người dùng ấn phím q hoặc phím ESC (mã 27), ngắt vòng lặp để tắt
            elif k==ord('s'): cv2.imwrite(f'data/{datetime.now():%Y%m%d_%H%M%S}.jpg',display)  # [Dòng 155] Nếu người dùng ấn nút s, chụp ảnh giao diện lưu vào thư mục data/
    finally:  # [Dòng 156] Không kể chương trình chạy bình thường hay bị lỗi, luôn luôn thực thi giải phóng camera ở đây
        cap.release(); cv2.destroyAllWindows()  # [Dòng 157] Tắt webcam camera và tắt toàn bộ các cửa sổ OpenCV trên màn hình
        stats.report()  # [Dòng 158] Gọi hàm report ở dòng 89 để tự động vẽ và xuất ra Dashboard reports/emotion_report.png

def mode_video(args):  # [Dòng 160] Định nghĩa hàm chạy nhận diện cảm xúc từ file video offline
    if not args.input or not os.path.isfile(args.input): print('Khong tim thay video!'); return  # [Dòng 161] Nếu không có video đầu vào, in lỗi và thoát
    cap = cv2.VideoCapture(args.input)  # [Dòng 162] Mở file video đầu vào bằng OpenCV
    eng = EmotionEngine(args.model)  # [Dòng 163] Khởi tạo bộ engine xử lý
    stats = SessionStats(); fc=0; skip=2; last=[]; start_t = time.time()  # [Dòng 164] Khởi tạo các thông số ghi nhận thống kê tương tự webcam
    fps = cap.get(cv2.CAP_PROP_FPS) or 25  # [Dòng 165] Lấy tốc độ khung hình (FPS) của video để đồng bộ thời gian ghi file đầu ra
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # [Dòng 166] Lấy tổng số khung hình của video dùng để tính toán thanh tiến trình (progress bar)
    writer = None  # [Dòng 167] Khởi tạo biến ghi video writer mặc định bằng None
    if args.output:  # [Dòng 168] Nếu người dùng truyền đường dẫn video đầu ra muốn lưu kết quả
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # [Dòng 169] Lấy chiều rộng và chiều cao đúng của video gốc
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))  # [Dòng 170] Khởi tạo bộ ghi VideoWriter MP4

    try:  # [Dòng 172] Khối block try an toàn để đọc video
        while True:  # [Dòng 173] Vòng lặp vô hạn đọc từng khung hình của video
            ret,frame = cap.read()  # [Dòng 174] Đọc frame tiếp theo của video
            if not ret: break  # [Dòng 175] Khi video hết khung hình (hết phim), tự động thoát vòng lặp
            fc+=1  # [Dòng 176] Tăng biến đếm khung hình của video thêm 1
            if fc%skip==0: last=eng.process(frame); stats.record(time.time()-start_t,last)  # [Dòng 177] Nhảy khung hình: chỉ xử lý AI ở frame chẵn để video chạy mượt, ghi nhận stats
            display = draw_boxes(frame, last)  # [Dòng 178] Vẽ nhãn cảm xúc lên frame hiện tại
            prog = fc/total if total>0 else 0  # [Dòng 179] Tính phần trăm tiến độ video hiện tại từ 0.0 đến 1.0
            cv2.rectangle(display,(0,display.shape[0]-6),(display.shape[1],display.shape[0]),(50,50,50),-1)  # [Dòng 180] Vẽ ô vuông xám đậm ở sát đáy màn hình làm đường ray cho thanh tiến trình
            cv2.rectangle(display,(0,display.shape[0]-6),(int(display.shape[1]*prog),display.shape[0]),(0,200,255),-1)  # [Dòng 181] Vẽ đè ô màu vàng dài bằng đúng prog x display.width để biểu thị thanh tiến độ chạy
            cv2.putText(display,f'Frame {fc}/{total} Faces:{len(last)}',(10,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)  # [Dòng 182] In thông tin tiến trình frame lên góc trái màn hình
            if writer: writer.write(display)  # [Dòng 183] Ghi ghi frame đã vẽ nhãn vào video đầu ra nếu có yêu cầu ghi file
            cv2.imshow('Emotion Detection',display)  # [Dòng 184] Hiển thị cửa sổ video lên màn hình
            k=cv2.waitKey(1)&0xFF  # [Dòng 185] Bắt phím bấm của người dùng chờ 1 mili giây
            if k in (ord('q'),27): break  # [Dòng 186] Nếu nhấn q hoặc ESC, dừng video ngay lập tức
            elif k==ord(' '):  # [Dòng 187] Nếu nhấn nút SPACE (Dấu cách)
                while True:  # [Dòng 188] Chạy vòng lặp vô hạn để dừng phim (Pause video)
                    if cv2.waitKey(100)&0xFF!=ord(' '): continue  # [Dòng 189] Nếu chưa bấm lại nút SPACE, tiếp tục treo
                    else: break  # [Dòng 190] Nếu bấm lại nút SPACE, tiếp tục chạy tiếp video
    finally:  # [Dòng 191] Khai báo khối finally đóng và giải phóng file video
        cap.release()  # [Dòng 192] Ngắt kết nối đọc video
        if writer: writer.release()  # [Dòng 193] Giải phóng file video đầu ra để ghi hoàn tất dữ liệu xuất ra đĩa
        cv2.destroyAllWindows()  # [Dòng 194] Đóng cửa sổ video trên màn hình
        stats.report()  # [Dòng 195] Gọi vẽ Dashboard emotion_report.png báo cáo cuối video

def mode_image(args):  # [Dòng 197] Định nghĩa hàm nhận diện cảm xúc trên file ảnh tĩnh
    if not args.input or not os.path.isfile(args.input): print('Khong tim thay anh!'); return  # [Dòng 198] Nếu không tìm thấy ảnh đầu vào, in lỗi và thoát
    frame = cv2.imread(args.input)  # [Dòng 199] Đọc file ảnh vào bộ nhớ bằng OpenCV
    if frame is None: print('Ko doc duoc anh!'); return  # [Dòng 200] Nếu ảnh lỗi không thể đọc được, in lỗi và thoát
    eng = EmotionEngine(args.model)  # [Dòng 201] Khởi tạo bộ engine xử lý
    results = eng.process(frame)  # [Dòng 202] Quét phát hiện và nhận dạng cảm xúc tất cả các mặt có trong ảnh
    display = draw_boxes(frame, results)  # [Dòng 203] Vẽ bounding box và viết nhãn lên ảnh
    if args.output: cv2.imwrite(args.output, display); print(f'Da luu: {args.output}')  # [Dòng 204] Nếu có tham số output, lưu ảnh kết quả xuống ổ cứng
    cv2.imshow('Emotion Detection',display)  # [Dòng 205] Mở cửa sổ hiển thị ảnh kết quả lên màn hình
    cv2.waitKey(0); cv2.destroyAllWindows()  # [Dòng 206] Treo cửa sổ vô hạn cho đến khi người dùng ấn 1 phím bất kỳ thì tắt ô

# ── Main ──  # [Dòng 208] Vùng định nghĩa điểm khởi chạy chương trình
if __name__ == '__main__':  # [Dòng 209] Điểm vào chính thức của chương trình Python chạy từ terminal
    p = argparse.ArgumentParser(description='Nhan dang cam xuc')  # [Dòng 210] Khởi tạo bộ nhận tham số dòng lệnh argparse
    p.add_argument('--mode', default='webcam', choices=['webcam','image','video'])  # [Dòng 211] Tham số --mode để chọn chế độ chạy: webcam (mặc định), image, video
    p.add_argument('--model', default='enet_b0_8_best_afew')  # [Dòng 212] Tham số --model để chọn model của HSEmotion. Mặc định dùng EfficientNet-B0
    p.add_argument('--camera', type=int, default=0)  # [Dòng 213] Tham số --camera để chọn cổng camera kết nối (mặc định là 0 - webcam gốc)
    p.add_argument('--input', default=None)  # [Dòng 214] Tham số --input nhận đường dẫn file video hoặc ảnh đầu vào
    p.add_argument('--output', default=None)  # [Dòng 215] Tham số --output nhận đường dẫn file ghi video hoặc ghi ảnh kết quả
    args = p.parse_args()  # [Dòng 216] Phân tích và nạp các tham số người dùng đã nhập vào biến đối tượng args
    {'webcam':mode_webcam,'image':mode_image,'video':mode_video}[args.mode](args)  # [Dòng 217] Kỹ thuật Dictionary Mapping gọi động hàm (mode_webcam/mode_image/mode_video) tương ứng, truyền tham số args
