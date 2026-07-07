# -*- coding: utf-8 -*-
"""
main.py - He thong nhan dang cam xuc nguoi hoc online
Su dung: python main.py [--mode webcam|image|video]
"""

import os, sys, time, argparse, numpy as np, cv2, torch  # [Dòng 6] Nạp các thư viện công cụ hỗ trợ (quản lý thư mục, đo thời gian học, mở webcam, vẽ khung mặt, chạy AI). Không có thư viện này máy tính sẽ không thể xử lý hình ảnh.
from collections import deque, Counter  # [Dòng 7] deque: tạo hộp nhớ 10 giây gần nhất để tránh nháy nhãn hiển thị. Counter: bộ đếm số lần xuất hiện cảm xúc để tính phần trăm vẽ biểu đồ tròn.
from datetime import datetime  # [Dòng 8] Lấy ngày giờ thực tế hiện tại để đặt tên file ảnh chụp màn hình khi người dùng bấm nút chụp ảnh (phím 's').
from hsemotion.facial_emotions import HSEmotionRecognizer  # [Dòng 9] Nạp lớp HSEmotionRecognizer chứa mô hình AI EfficientNet-B0 pretrained (đã học sẵn từ 450.000 khuôn mặt trên thế giới).
import matplotlib.pyplot as plt  # [Dòng 10] Nạp thư viện vẽ biểu đồ tròn và dòng thời gian cảm xúc khi kết thúc buổi học để giáo viên đánh giá.

# ── Ghi chú: Khai báo các hằng số cấu hình toàn cục của hệ thống. ──
EMOTIONS = ['Anger','Disgust','Happiness','Neutral','Sadness','Surprise']  # [Dòng 13] Danh sách 6 cảm xúc đích nhận diện. Bỏ đi 2 lớp không dùng trong học tập trực tuyến là Sợ Hãi và Khinh Bỉ.
EMO_COLORS = {'Happy':(0,255,0),'Sad':(255,80,80),'Angry':(0,0,255),  # [Dòng 14] Đặt màu sắc vẽ bounding box: OpenCV mặc định thứ tự ngược BGR. Happy vẽ Xanh lá (0,255,0), Sad vẽ Xanh dương nhạt, Angry vẽ Đỏ.
              'Surprise':(0,165,255),'Neutral':(180,180,180),'Disgust':(0,160,160)}  # [Dòng 15] Tiếp tục mã màu: Surprise vẽ màu Cam, Neutral vẽ màu Xám nền, Disgust vẽ màu Vàng đất.
DISPLAY = {'Anger':'Angry','Happiness':'Happy','Sadness':'Sad','Contempt':'Disgust'}  # [Dòng 16] Đổi tên nhãn sang nhãn ngắn gọn, thân thiện để hiển thị đẹp mắt trên giao diện camera.
FACE_CASCADE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # [Dòng 17] Nạp thuật toán quét tìm vị trí khuôn mặt Haar Cascade tích hợp sẵn trong thư viện OpenCV.

# ── Ghi chú: Khai báo Class EmotionEngine - bộ máy xử lý kỹ thuật, tiền xử lý và chạy mô hình dự đoán. ──
class EmotionEngine:  # [Dòng 20] Tạo lớp đối tượng xử lý: Quét tìm khuôn mặt, tiền xử lý CLAHE, đoán cảm xúc bằng AI và bám vết người học.
    def __init__(self, model_name='enet_b0_8_best_afew'):  # [Dòng 21] Hàm chạy các cài đặt ban đầu khi khởi động bộ máy EmotionEngine.
  # [Dòng 22] Ghi chú kỹ thuật tắt cảnh báo PyTorch khi chạy trên máy.
        torch_load = torch.load  # [Dòng 23] Lưu tạm hàm nạp model gốc của PyTorch.
        torch.load = lambda f,**kw: torch_load(f,**{**kw,'weights_only':False})  # [Dòng 24] Ghi đè hàm load bằng lambda để ép weights_only=False, tránh các dòng cảnh báo bảo mật màu đỏ làm bẩn màn hình.
        self.er = HSEmotionRecognizer(model_name=model_name)  # [Dòng 25] Tải mô hình AI EfficientNet-B0 pretrained (~25MB) lên RAM máy tính. Chọn B0 vì siêu nhẹ, chạy mượt trên CPU.
        torch.load = torch_load  # [Dòng 26] Khôi phục lại hàm load gốc tránh làm ảnh hưởng các phần code khác.
        
        self.fc = cv2.CascadeClassifier(FACE_CASCADE)  # [Dòng 28] Kích hoạt bộ quét tìm khuôn mặt từ file XML cấu hình đã nạp ở dòng 17.
        self.bufs = {}  # [Dòng 29] Từ điển lưu trữ bộ đệm xác suất của từng người học để phục vụ làm mịn nhãn hiển thị.
        self.BSZ, self.GRID = 10, 60  # [Dòng 30] BSZ=10: nhớ 10 khung hình (0.3s) gần nhất để tính trung bình. GRID=60: nếu tâm mặt nhúc nhích lệch dưới 60px thì vẫn coi là một người.
        self.THRESH = {'Happiness':.3,'Sadness':.2,'Surprise':.3,'Anger':.3,'Neutral':.35,'Disgust':.3}  # [Dòng 31] Bộ ngưỡng nhạy thích nghi: Happiness cần 30% để tránh nhiễu khẩu hình miệng, Sadness chỉ cần 20% để tăng độ nhạy.
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # [Dòng 32] Cân bằng sáng cục bộ: chia mặt thành lưới 8x8 ô làm sáng đều. clipLimit=2.0 để giới hạn tương phản, tránh bị nhiễu hạt khi thiếu sáng.

    def detect(self, frame):  # [Dòng 34] Định nghĩa hàm tìm tọa độ các khuôn mặt trên khung hình camera.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # [Dòng 35] Chuyển khung hình sang ảnh xám Grayscale vì tìm mặt chỉ cần thông tin sáng/tối, giúp máy chạy nhanh hơn.
        return self.fc.detectMultiScale(gray, 1.1, 5, minSize=(30,30))  # [Dòng 36] scaleFactor=1.1: thu nhỏ ảnh 10% mỗi lượt quét để bắt được cả mặt to và nhỏ. minNeighbors=5: ô quét phải trùng lặp 5 lần để tránh nhận nhầm đồ vật.

    def predict(self, face_img):  # [Dòng 38] Định nghĩa hàm tính toán cảm xúc cho một khuôn mặt đã cắt ra.
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)  # [Dòng 39] Chuyển ảnh cắt khuôn mặt sang ảnh xám Grayscale để chuẩn bị cân bằng sáng.
        eq = self.clahe.apply(gray)  # [Dòng 40] Áp dụng CLAHE làm sáng đều vùng mặt bị ngược sáng hoặc bóng đổ, giúp các nếp nhăn biểu cảm hiện rõ rệt.
        rgb = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)  # [Dòng 41] Chuyển lại sang ảnh màu RGB vì bộ não AI yêu cầu ảnh đầu vào phải có 3 kênh màu.
        name, scores = self.er.predict_emotions(rgb, logits=False)  # [Dòng 42] Đưa ảnh mặt vào AI để lấy về điểm số phần trăm của 8 cảm xúc gốc.
  # [Dòng 43] Ghi chú: Phép ánh xạ 8 cảm xúc gốc sang 6 cảm xúc đích.
        raw6 = np.array([scores[0], scores[2], scores[4], scores[5], scores[6], scores[7]])  # [Dòng 44] Lọc lấy 6 cảm xúc cần nhận dạng, bỏ đi Sợ Hãi và Khinh Bỉ.
        scores = raw6 / raw6.sum() if raw6.sum() > 0 else raw6  # [Dòng 45] Softmax rút gọn: chia điểm từng lớp cho tổng 6 lớp để tổng phần trăm quay về đúng 100% hợp lệ toán học.
        return scores  # [Dòng 46] Trả về danh sách xác suất 6 cảm xúc đã được chuẩn hóa.

    def process(self, frame):  # [Dòng 48] Hàm tổng xử lý chính trên khung hình camera (tìm mặt -> bám vết người học -> đoán cảm xúc -> làm mịn -> lọc ngưỡng).
        results = []  # [Dòng 49] Khởi tạo danh sách trống chứa kết quả của các khuôn mặt trong khung hình hiện tại.
        faces = self.detect(frame)  # [Dòng 50] Gọi hàm detect ở dòng 34 để tìm tất cả các khuôn mặt có trên camera.
        if len(faces) > 1:  # [Dòng 51] Nếu camera quét thấy nhiều hơn 1 người học.
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)  # [Dòng 52] Sắp xếp khuôn mặt to nhất lên đầu để ưu tiên học sinh chính ngồi trực diện camera, bỏ qua người lạ đi qua phía sau.
        active = set()  # [Dòng 53] Tạo danh sách tạm chứa ID các khuôn mặt đang xuất hiện thực tế.
        for (x,y,w,h) in faces:  # [Dòng 54] Vòng lặp duyệt qua từng khuôn mặt phát hiện được.
            if w>=48 and h>=48:  # [Dòng 55] Bỏ qua các khuôn mặt quá nhỏ hoặc ở quá xa camera (yêu cầu kích thước tối thiểu 48x48 pixel).
                k = ((x+w//2)//self.GRID*self.GRID, (y+h//2)//self.GRID*self.GRID)  # [Dòng 56] Centroid Tracking: Lượng tử hóa tâm mặt về ô lưới 60px làm ID bám vết khuôn mặt để lấy đúng bộ đệm lịch sử.
                active.add(k)  # [Dòng 57] Ghi nhận ID khuôn mặt này đang hoạt động thực tế trên khung hình.
        for k in set(self.bufs.keys())-active:  # [Dòng 58] Vòng lặp tìm các khuôn mặt có trong bộ đệm nhưng không còn xuất hiện trên camera nữa.
            del self.bufs[k]  # [Dòng 59] Xóa bộ đệm của họ đi để giải phóng RAM cho máy tính chạy nhẹ, tránh rò rỉ bộ nhớ.
        for (x,y,w,h) in faces:  # [Dòng 60] Vòng lặp thứ hai để tiến hành đoán cảm xúc cho từng người học.
            if w<48 or h<48: continue  # [Dòng 61] Bỏ qua các khuôn mặt ở xa quá nhỏ không đủ độ phân giải biểu cảm.
            k = ((x+w//2)//self.GRID*self.GRID, (y+h//2)//self.GRID*self.GRID)  # [Dòng 62] Xác định ID tương ứng của khuôn mặt để lấy đúng bộ nhớ lịch sử.
            face = frame[y:y+h,x:x+w]  # [Dòng 63] Cắt lấy riêng vùng ảnh chứa khuôn mặt từ khung hình camera gốc.
            scores = self.predict(face)  # [Dòng 64] Gọi hàm predict ở dòng 38 để AI tính toán phần trăm cảm xúc của khuôn mặt này.
            if k not in self.bufs: self.bufs[k] = deque(maxlen=self.BSZ)  # [Dòng 65] Nếu là người mới xuất hiện, tạo một hàng đợi lưu tối đa 10 khung hình gần nhất.
            self.bufs[k].append(scores)  # [Dòng 66] Đẩy kết quả phần trăm mới dự đoán vào hàng đợi bộ nhớ.
            avg = np.mean(self.bufs[k], axis=0)  # [Dòng 67] Lấy trung bình cộng xác suất của 10 khung hình để nhãn hiển thị được mượt mà, không bị nhấp nháy liên tục khi học sinh chớp mắt.
            idx = np.argmax(avg)  # [Dòng 68] Tìm cảm xúc có điểm số trung bình lớn nhất.
            conf = float(avg[idx])  # [Dòng 69] Lấy điểm tin cậy (phần trăm) của cảm xúc lớn nhất đó.
            emo = EMOTIONS[idx]  # [Dòng 70] Lấy tên cảm xúc tương ứng.
            if conf < self.THRESH.get(emo, .4):  # [Dòng 71] Lọc qua bộ ngưỡng thích nghi: nếu độ tin cậy thấp hơn ngưỡng quy định ở dòng 31.
                emo, conf = 'Neutral', float(avg[EMOTIONS.index('Neutral')])  # [Dòng 72] Ép cảm xúc về nhãn mặc định Neutral (Bình thường) để tránh đoán bừa bãi khi biểu cảm không rõ ràng.
            else:  # [Dòng 73] Trường hợp đạt trên ngưỡng thích nghi.
                emo = DISPLAY.get(emo, emo)  # [Dòng 74] Đổi tên nhãn sang nhãn ngắn gọn hiển thị (ví dụ: Happiness đổi thành Happy).
            results.append({'bbox':(x,y,w,h),'emotion':emo,'confidence':conf,'probs':avg})  # [Dòng 75] Đóng gói tọa độ khung mặt, tên cảm xúc và độ tin cậy vào danh sách.
        return results  # [Dòng 76] Trả về danh sách kết quả xử lý của toàn bộ các khuôn mặt trong khung hình này.

# ── Ghi chú: Khai báo các biến và lớp lưu trữ thông tin thống kê. ──
class SessionStats:  # [Dòng 79] Tạo đối tượng ghi nhận lịch sử biểu cảm và vẽ biểu đồ báo cáo khi đóng ứng dụng.
    def __init__(self):  # [Dòng 80] Cấu hình cài đặt ban đầu cho bộ thống kê.
        self.history = []  # [Dòng 81] Tạo một danh sách trống trong RAM để lưu trữ dòng thời gian và cảm xúc của học sinh.
        self.order = list(EMO_COLORS.keys())  # [Dòng 82] Sắp xếp thứ tự các cảm xúc trên trục đứng Y của đồ thị dòng thời gian.

    def record(self, t, results):  # [Dòng 84] Hàm ghi nhận cảm xúc của học sinh tại giây học thứ t.
        if results:  # [Dòng 85] Nếu camera quét thấy có người học.
            r = results[0]  # [Dòng 86] Chỉ lấy dữ liệu của học sinh chính ngồi gần nhất (results[0]), bỏ qua người lạ đi qua phía sau.
            self.history.append({'time':t,'emotion':r['emotion'],'confidence':r['confidence']})  # [Dòng 87] Lưu giây học, tên cảm xúc và độ tin cậy vào RAM (độ phức tạp O(1) rất nhanh).

    def report(self, out_dir='reports'):  # [Dòng 89] Hàm tự động vẽ và lưu Dashboard reports/emotion_report.png khi buổi học kết thúc.
        if not self.history: return  # [Dòng 90] Nếu chưa ghi nhận được dữ liệu nào (ví dụ vừa mở lên đã tắt ngay), dừng hàm không vẽ.
        os.makedirs(out_dir, exist_ok=True)  # [Dòng 91] Tạo thư mục reports/ lưu báo cáo trên máy tính nếu chưa có sẵn.
        ts = [h['time'] for h in self.history]  # [Dòng 92] Tách lấy danh sách mốc thời gian (giây) từ lịch sử.
        emos = [h['emotion'] for h in self.history]  # [Dòng 93] Tách lấy danh sách tên cảm xúc tương ứng.
        confs = [h['confidence'] for h in self.history]  # [Dòng 94] Tách lấy danh sách độ tin cậy.
        cnt = Counter(emos)  # [Dòng 95] Đếm số lần xuất hiện của từng cảm xúc để tính toán phần trăm cho biểu đồ tròn.
        present = [e for e in self.order if e in cnt]  # [Dòng 96] Lọc các cảm xúc thực tế có xuất hiện trong giờ học.
        counts = [cnt[e] for e in present]  # [Dòng 97] Lấy số lượng đếm tương ứng của các cảm xúc đó.
        colors = [tuple(c/255 for c in reversed(EMO_COLORS[e])) for e in present]  # [Dòng 98] Chuyển đổi mã màu BGR sang tỉ lệ RGB (0.0 - 1.0) của Matplotlib.
        emo2y = {e:i for i,e in enumerate(self.order)}  # [Dòng 99] Ánh xạ tên cảm xúc sang tọa độ số nguyên 0-5 làm mốc vẽ đồ thị Y.
        yv = [emo2y[e] for e in emos]  # [Dòng 100] Tạo mảng các tọa độ Y tương ứng với lịch sử cảm xúc.

        fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12,10),gridspec_kw={'height_ratios':[1,1.5]})  # [Dòng 102] Tạo khung vẽ rộng 12x10 inch, chia dọc thành 2 đồ thị: ax1 (biểu đồ tròn ở trên), ax2 (timeline ở dưới).
        ax1.pie(counts, labels=present, colors=colors, autopct=lambda p:f'{p:.1f}%', startangle=90)  # [Dòng 103] Vẽ biểu đồ hình tròn (Pie Chart) thể hiện phần trăm phân bố thời gian của các cảm xúc.
        ax1.set_title('Phan Bo Cam Xuc')  # [Dòng 104] Đặt tiêu đề cho biểu đồ tròn phía trên là 'Phan Bo Cam Xuc'.
        ax2.step(ts, yv, where='post', color='gray', alpha=.5, linewidth=1.5, linestyle='--')  # [Dòng 105] Vẽ đường nét đứt bậc thang màu xám kết nối các mốc cảm xúc theo thời gian.
        for t,e,c in zip(ts,emos,confs):  # [Dòng 106] Vòng lặp vẽ các chấm tròn màu hiển thị cảm xúc.
            rgb = tuple(v/255 for v in reversed(EMO_COLORS[e]))  # [Dòng 107] Xác định màu sắc phù hợp cho chấm tròn theo đúng loại cảm xúc.
            ax2.scatter(t, emo2y[e], s=max(40,c*250), color=[rgb], edgecolors='black', linewidth=.5)  # [Dòng 108] Vẽ chấm tròn màu. Độ tin cậy càng cao chấm tròn càng to: size = max(40, độ tin cậy * 250).
        ax2.set_yticks(range(len(self.order)))  # [Dòng 109] Đặt vạch chia tọa độ trên trục Y từ 0 đến 5.
        ax2.set_yticklabels(self.order)  # [Dòng 110] Ghi chữ tên các cảm xúc tương ứng vào trục đứng Y.
        ax2.set_xlabel('Thoi gian (giay)')  # [Dòng 111] Đặt nhãn trục ngang X là 'Thời gian (giây)'.
        ax2.set_ylabel('Cam xuc')  # [Dòng 112] Đặt nhãn trục đứng Y là 'Cảm xúc'.
        ax2.grid(True, alpha=.3)  # [Dòng 113] Bật lưới ô tọa độ mờ (độ trong suốt 0.3) để dễ nhìn dòng gióng.
        plt.tight_layout()  # [Dòng 114] Tự động căn chỉnh lề các biểu đồ để chữ không bị đè và hình hiển thị cân đối.
        plt.savefig(os.path.join(out_dir,'emotion_report.png'), dpi=150, bbox_inches='tight')  # [Dòng 115] Lưu toàn bộ Dashboard thành file ảnh reports/emotion_report.png chất lượng cao (150 DPI).
        plt.close()  # [Dòng 116] Giải phóng bộ nhớ RAM bằng cách đóng hình vẽ đồ thị sau khi đã ghi file xong.
        print(f'  Da luu: reports/emotion_report.png')  # [Dòng 117] In dòng thông báo báo cáo đã lưu thành công ra màn hình đen console.

# ── Ghi chú: Khai báo các hàm vẽ giao diện đồ họa phụ trợ. ──
def draw_boxes(frame, results):  # [Dòng 120] Định nghĩa hàm draw_boxes nhận khung hình camera thô và kết quả kết xuất.
    out = frame.copy()  # [Dòng 121] Tạo bản sao của ảnh để vẽ, tránh vẽ trực tiếp làm hỏng ma trận ảnh gốc của webcam.
    for r in results:  # [Dòng 122] Vòng lặp duyệt qua thông tin nhận diện của từng khuôn mặt.
        x,y,w,h = r['bbox']; emo = r['emotion']; conf = r['confidence']  # [Dòng 123] Trích xuất tọa độ khung mặt (x,y,w,h), tên cảm xúc và độ tin cậy.
        color = EMO_COLORS.get(emo,(255,255,255))  # [Dòng 124] Lấy màu sắc vẽ khung tương ứng với cảm xúc đó (mặc định dùng màu trắng nếu không có).
        cv2.rectangle(out,(x,y),(x+w,y+h),color,2)  # [Dòng 125] Vẽ khung hình chữ nhật nét dày 2px bao quanh khuôn mặt học sinh.
        lbl = f'{emo}: {conf:.0%}'  # [Dòng 126] Tạo nhãn hiển thị gồm tên cảm xúc và phần trăm độ tin cậy được làm tròn (ví dụ: Happy: 98%).
        (lw,lh),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,.65,2)  # [Dòng 127] Đo chiều cao và rộng của chữ nhãn để thiết kế hộp nền đè vừa khít.
        cv2.rectangle(out,(x,y-lh-10),(x+lw+6,y),color,-1)  # [Dòng 128] Vẽ một ô chữ nhật đặc (độ dày nét -1) làm nền nhãn ngay phía trên khung mặt để nhãn hiển thị rõ ràng, không bị chìm vào hậu cảnh.
        cv2.putText(out,lbl,(x+3,y-5),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),2)  # [Dòng 129] Viết nhãn chữ màu đen (0,0,0) đè lên trên ô hộp nền đặc nét chữ dày 2px.
    return out  # [Dòng 130] Trả về khung hình camera đã được vẽ trang trí giao diện hoàn thiện.

# ── Ghi chú: Các chế độ hoạt động chính của hệ thống. ──
def mode_webcam(args):  # [Dòng 133] Hàm chạy chế độ Webcam trực tiếp thời gian thực.
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # [Dòng 134] Mở camera bằng DirectShow để camera khởi động nhanh, không bị trễ nạp driver trên Windows.
    if not cap.isOpened(): cap = cv2.VideoCapture(args.camera)  # [Dòng 135] Nếu lỗi DirectShow, tự động nạp webcam bằng hàm mặc định của OpenCV.
    if not cap.isOpened(): print('Ko mo duoc camera!'); return  # [Dòng 136] Nếu vẫn không mở được camera (bị ứng dụng khác chiếm dụng), báo lỗi và tắt.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)  # [Dòng 137] Thiết lập độ phân giải của camera chuẩn là 640x480 pixel.
    eng = EmotionEngine(args.model)  # [Dòng 138] Khởi tạo bộ máy xử lý cảm xúc với model EfficientNet đã chọn.
    stats = SessionStats(); fc=0; skip=2; last=[]; start_t = time.time()  # [Dòng 139] Thiết lập bộ thống kê stats, biến đếm frame fc, bước nhảy frame skip=2 để giảm tải 50% CPU cho máy chạy mát, mốc thời gian bắt đầu.

    os.makedirs('data', exist_ok=True)  # [Dòng 141] Tự động tạo thư mục data/ dùng để chứa các ảnh chụp màn hình khi người học chụp.

    try:  # [Dòng 143] Khối lệnh try an toàn giúp giải phóng webcam kể cả khi có lỗi tắt đột ngột xảy ra.
        while True:  # [Dòng 144] Vòng lặp chạy vô tận để liên tục đọc ảnh webcam.
            ret,frame = cap.read()  # [Dòng 145] Đọc một khung hình thô từ webcam. ret báo đọc thành công hay không, frame chứa ảnh.
            if not ret: break  # [Dòng 146] Nếu webcam bị mất kết nối hoặc không đọc được ảnh, tự động thoát vòng lặp.
            fc+=1  # [Dòng 147] Tăng biến đếm số lượng khung hình thêm 1 đơn vị.
            if fc%skip==0: last=eng.process(frame); stats.record(time.time()-start_t,last)  # [Dòng 148] Chỉ chạy AI xử lý ở khung hình chẵn (2,4,6...) để giảm tải 50% CPU cho máy chạy mát, ghi nhận stats cảm xúc.
            display = draw_boxes(frame, last)  # [Dòng 149] Vẽ bounding box viền màu sắc và nhãn cảm xúc lên khung hình hiện tại.
            cv2.putText(display,f'Frame:{fc} Faces:{len(last)}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)  # [Dòng 150] Ghi thông số frame và số mặt phát hiện lên góc trên bên trái màn hình.
            cv2.putText(display,'q:thoat s:chup',(10,display.shape[0]-12),cv2.FONT_HERSHEY_SIMPLEX,.45,(200,200,200),1)  # [Dòng 151] Ghi dòng chữ hướng dẫn phím tắt dưới đáy màn hình bên trái.
            cv2.imshow('Emotion Detection',display)  # [Dòng 152] Hiển thị cửa sổ camera nhận diện cảm xúc lên màn hình máy tính.
            k=cv2.waitKey(1)&0xFF  # [Dòng 153] Đọc phím bấm từ người dùng với thời gian chờ phản hồi là 1 mili giây.
            if k in (ord('q'),27): break  # [Dòng 154] Nếu người dùng nhấn phím q hoặc nút ESC, thoát vòng lặp để đóng camera.
            elif k==ord('s'): cv2.imwrite(f'data/{datetime.now():%Y%m%d_%H%M%S}.jpg',display)  # [Dòng 155] Nếu người dùng nhấn phím s, chụp ảnh màn hình lưu vào data/ dạng tệp JPG theo ngày giờ.
    finally:  # [Dòng 156] Dù chương trình chạy mượt hay gặp lỗi tắt đột ngột, khối này luôn được chạy để bảo vệ webcam.
        cap.release(); cv2.destroyAllWindows()  # [Dòng 157] Tắt camera webcam và đóng toàn bộ các cửa sổ hiển thị trên màn hình.
        stats.report()  # [Dòng 158] Tự động xuất biểu đồ Dashboard báo cáo lưu vào reports/emotion_report.png.

def mode_video(args):  # [Dòng 160] Hàm chạy chế độ đọc tệp video offline có sẵn.
    if not args.input or not os.path.isfile(args.input): print('Khong tim thay video!'); return  # [Dòng 161] Nếu đường dẫn trống hoặc file video không tồn tại, in lỗi và kết thúc.
    cap = cv2.VideoCapture(args.input)  # [Dòng 162] Mở file video bằng thư viện OpenCV.
    eng = EmotionEngine(args.model)  # [Dòng 163] Khởi tạo bộ máy xử lý cảm xúc.
    stats = SessionStats(); fc=0; skip=2; last=[]; start_t = time.time()  # [Dòng 164] Khởi tạo các thông số đếm frame và ghi nhận stats tương tự webcam.
    fps = cap.get(cv2.CAP_PROP_FPS) or 25  # [Dòng 165] Lấy thông số tốc độ khung hình (FPS) của video để ghi video kết quả đầu ra khớp tốc độ.
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # [Dòng 166] Lấy tổng số lượng khung hình của tệp video để tính tiến độ chạy.
    writer = None  # [Dòng 167] Khởi tạo biến ghi video mặc định là trống (None).
    if args.output:  # [Dòng 168] Nếu người dùng yêu cầu lưu lại tệp video kết quả đầu ra sau khi chạy.
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # [Dòng 169] Lấy chính xác chiều rộng và chiều cao của video gốc.
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))  # [Dòng 170] Khởi tạo công cụ ghi file video định dạng MP4.

    try:
        while True:  # [Dòng 173] Vòng lặp vô hạn đọc từng khung hình video tuần tự.
            ret,frame = cap.read()  # [Dòng 174] Đọc khung hình video tiếp theo.
            if not ret: break  # [Dòng 175] Khi video đã chạy hết (hết khung hình), thoát khỏi vòng lặp.
            fc+=1  # [Dòng 176] Tăng biến đếm khung hình của video thêm 1.
            if fc%skip==0: last=eng.process(frame); stats.record(time.time()-start_t,last)  # [Dòng 177] Chỉ chạy AI xử lý ở khung hình chẵn để video chạy nhanh, ghi nhận stats cảm xúc vào RAM.
            display = draw_boxes(frame, last)  # [Dòng 178] Vẽ nhãn cảm xúc và bounding box lên khung hình video hiện tại.
            prog = fc/total if total>0 else 0  # [Dòng 179] Tính phần trăm tiến độ xử lý video hiện tại (từ 0.0 đến 1.0).
            cv2.rectangle(display,(0,display.shape[0]-6),(display.shape[1],display.shape[0]),(50,50,50),-1)  # [Dòng 180] Vẽ một dải hộp chữ nhật đặc màu xám đậm dày 6px chạy dọc đáy màn hình làm nền thanh tiến trình.
            cv2.rectangle(display,(0,display.shape[0]-6),(int(display.shape[1]*prog),display.shape[0]),(0,200,255),-1)  # [Dòng 181] Vẽ dải màu vàng sáng biểu thị tiến độ thực tế đã chạy của video (prog * display_width).
            cv2.putText(display,f'Frame {fc}/{total} Faces:{len(last)}',(10,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)  # [Dòng 182] Ghi tiến độ khung hình video hiện tại lên góc trên bên trái màn hình.
            if writer: writer.write(display)  # [Dòng 183] Ghi khung hình đã được vẽ nhãn cảm xúc vào tệp video đầu ra nếu cấu hình.
            cv2.imshow('Emotion Detection',display)  # [Dòng 184] Hiển thị cửa sổ video nhận diện cảm xúc lên màn hình.
            k=cv2.waitKey(1)&0xFF  # [Dòng 185] Bắt phím nhấn của người dùng chờ 1 mili giây.
            if k in (ord('q'),27): break  # [Dòng 186] Nếu nhấn q hoặc ESC, dừng video và thoát ngay lập tức.
            elif k==ord(' '):  # [Dòng 187] Nếu nhấn phím cách (SPACE).
                while True:  # [Dòng 188] Khởi chạy vòng lặp vô hạn đứng yên để tạm dừng (Pause) video.
                    if cv2.waitKey(100)&0xFF!=ord(' '): continue  # [Dòng 189] Nếu chưa nhấn lại nút SPACE, tiếp tục treo video đứng yên.
                    else: break  # [Dòng 190] Nếu nhấn lại nút SPACE, thoát vòng lặp treo để video chạy tiếp (Resume).
    finally:  # [Dòng 191] Khối finally đóng và giải phóng file video.
        cap.release()  # [Dòng 192] Tắt kết nối đọc file video.
        if writer: writer.release()  # [Dòng 193] Đóng file ghi kết quả đầu ra để ghi hoàn tất dữ liệu lên ổ cứng.
        cv2.destroyAllWindows()  # [Dòng 194] Đóng toàn bộ cửa sổ hiển thị video trên màn hình.
        stats.report()  # [Dòng 195] Tự động xuất biểu đồ Dashboard báo cáo lưu vào reports/emotion_report.png.

def mode_image(args):   # [Dòng 197] Hàm chạy chế độ nhận dạng trên một file ảnh tĩnh duy nhất.
    if not args.input or not os.path.isfile(args.input): print('Khong tim thay anh!'); return  # [Dòng 198] Nếu không tìm thấy file ảnh đầu vào, báo lỗi và kết thúc.
    frame = cv2.imread(args.input)  # [Dòng 199] Đọc ma trận điểm ảnh màu từ file ảnh đầu vào bằng OpenCV.
    if frame is None: print('Ko doc duoc anh!'); return  # [Dòng 200] Nếu ảnh lỗi không thể đọc được, in lỗi và kết thúc.
    eng = EmotionEngine(args.model)  # [Dòng 201] Khởi tạo bộ máy xử lý cảm xúc.
    results = eng.process(frame)   # [Dòng 202] Quét phát hiện và nhận dạng cảm xúc tất cả khuôn mặt trong ảnh.
    display = draw_boxes(frame, results)  # [Dòng 203] Vẽ bounding box viền màu sắc và nhãn cảm xúc lên ảnh.
    if args.output: cv2.imwrite(args.output, display); print(f'Da luu: {args.output}')  # [Dòng 204] Lưu ảnh kết quả xuống ổ cứng nếu có tham số output.
    cv2.imshow('Emotion Detection',display)  # [Dòng 205] Mở cửa sổ hiển thị ảnh kết quả lên màn hình máy tính.
    cv2.waitKey(0); cv2.destroyAllWindows()  # [Dòng 206] Treo màn hình vô hạn (đợi tham số 0) để người dùng xem kết quả, tự động tắt cửa sổ khi ấn phím bất kỳ.

# ── Ghi chú: Điểm khởi chạy chính của chương trình. ──
if __name__ == '__main__':  # [Dòng 209] Điểm khởi chạy chính thức của chương trình Python từ terminal dòng lệnh.
    p = argparse.ArgumentParser(description='Nhan dang cam xuc')  # [Dòng 210] Khởi tạo bộ đọc tham số dòng lệnh từ phía người dùng.
    p.add_argument('--mode', default='webcam', choices=['webcam','image','video'])  # [Dòng 211] Tham số --mode để chọn chế độ chạy: webcam (mặc định), image, video.
    p.add_argument('--model', default='enet_b0_8_best_afew')  # [Dòng 212] Tham số --model để chọn mô hình EfficientNet-B0 của HSEmotion.
    p.add_argument('--camera', type=int, default=0)  # [Dòng 213] Tham số --camera để chọn chỉ số cổng webcam vật lý (mặc định là cổng 0).
    p.add_argument('--input', default=None)  # [Dòng 214] Tham số --input nhận đường dẫn file video hoặc ảnh đầu vào khi chạy chế độ video/image.
    p.add_argument('--output', default=None)  # [Dòng 215] Tham số --output nhận đường dẫn lưu trữ file ghi video hoặc ảnh kết quả.
    args = p.parse_args()  # [Dòng 216] Phân tích và nạp toàn bộ tham số người dùng nhập vào biến args.
    {'webcam':mode_webcam,'image':mode_image,'video':mode_video}[args.mode](args)  # [Dòng 217] Sử dụng kỹ thuật Ánh xạ Từ điển để gọi hàm chạy chế độ tương ứng, truyền args làm tham số đầu vào.
