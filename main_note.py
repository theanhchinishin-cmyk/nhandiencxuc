"""
main_note.py - Phien ban ghi chu chi tiet tung dong 1 cua He thong nhan dang cam xuc nguoi hoc online
Su dung: python main_note.py [--mode webcam|image|video]
"""

import os, sys, time, argparse, numpy as np, cv2, torch  # [Dòng 6] Nap thu vien: os (quan ly file/thu muc), sys (bien he thong), time (do thoi gian), argparse (doc tham so terminal), numpy (ma tran), cv2 (OpenCV), torch (PyTorch)
from collections import deque, Counter  # [Dòng 7] deque: hang doi truot 10 khung hinh lam min nhan, Counter: bo dem so lan xuat hien cam xuc de ve bieu do tron
from datetime import datetime  # [Dòng 8] Lay ngay gio he thong thuc te de dat ten file anh chup man hinh khi nguoi dung an nut 's'
from hsemotion.facial_emotions import HSEmotionRecognizer  # [Dòng 9] Nap lop HSEmotionRecognizer chua mo hinh EfficientNet-B0 pretrained de du doan cam xuc
import matplotlib.pyplot as plt  # [Dòng 10] Nap thu vien ve do thi Matplotlib de tu dong xuat ra Dashboard báo cao PNG sau khi hoc xong

# ── Constants ──  # [Dong 12] Khai bao cac hang so cau hinh toan cuc cua he thong
EMOTIONS = ['Anger','Disgust','Happiness','Neutral','Sadness','Surprise']  # [Dong 13] Danh sach 6 nhan cam xuc dich ma he thong se phan loai va ve len man hinh
EMO_COLORS = {'Happy':(0,255,0),'Sad':(255,80,80),'Angry':(0,0,255),  # [Dong 14] Ban do mau BGR ve bounding box: Happy (Xanh la), Sad (Xanh duong nhat), Angry (Do)
              'Surprise':(0,165,255),'Neutral':(180,180,180),'Disgust':(0,160,160)}  # [Dong 15] Tiep tuc ma mau BGR: Surprise (Cam), Neutral (Xam), Disgust (Vang dat)
DISPLAY = {'Anger':'Angry','Happiness':'Happy','Sadness':'Sad','Contempt':'Disgust'}  # [Dong 16] Anh xa doi ten nhan noi bo sang ten hien thi ngan gon, dep mat tren giao dien camera
FACE_CASCADE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # [Dong 17] Lay file XML cau hinh Haar Cascade dung de quet va tim vi tri mat cua OpenCV

# ── Emotion Engine ──  # [Dong 19] Khai bao Class EmotionEngine - bo xu ly ki thuat, tien xu ly va chay mo hinh du doan
class EmotionEngine:  # [Dong 20] Dinh nghia lop EmotionEngine
    def __init__(self, model_name='enet_b0_8_best_afew'):  # [Dong 21] Ham khoi tao các thanh phan cua engine khi doi tuong duoc tao
        # Override torch.load to bypass weights_only warning in HSEmotionRecognizer  # [Dong 22] Chu thich ky thuat tat canh bao PyTorch
        torch_load = torch.load  # [Dong 23] Luu tam ham torch.load goc cua thu vien PyTorch vao bien
        torch.load = lambda f,**kw: torch_load(f,**{**kw,'weights_only':False})  # [Dong 24] Ghi de ham load bang lambda de ep weights_only=False, tat canh bao bao mat
        self.er = HSEmotionRecognizer(model_name=model_name)  # [Dong 25] Khoi tao mo hinh HSEmotion (EfficientNet-B0 pretrained ~25MB)
        torch.load = torch_load  # [Dong 26] Khoi phuc lai ham torch.load goc cua PyTorch ngay lap tuc de khong anh huong thu vien khac
        
        self.fc = cv2.CascadeClassifier(FACE_CASCADE)  # [Dong 28] Nap bo quet mat Haar Cascade tu file XML da dinh nghia o dong 17
        self.bufs = {}  # [Dong 29] Dict bufs: tu dien luu tru hang doi truot deque loc min xac suat cho tung khuon mat
        self.BSZ, self.GRID = 10, 60  # [Dong 30] BSZ = 10 (co hang doi lam min), GRID = 60 (kich thuoc o luoi de tracking tam mat)
        self.THRESH = {'Happiness':.3,'Sadness':.2,'Surprise':.3,'Anger':.3,'Neutral':.35,'Disgust':.3}  # [Dong 31] Bo nguong thich nghi de loc nhan, tranh bao gia
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # [Dong 32] Khoi tao CLAHE can bang sang cuc bo: clip limit = 2.0, chia luoi 8x8 o

    def detect(self, frame):  # [Dong 34] Dinh nghia ham phat hien cac khuon mat co trong mot khung hinh cap vao
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # [Dong 35] Chuyen anh mau BGR sang anh xam vi Haar Cascade chi can dung cuong do sang de quet
        return self.fc.detectMultiScale(gray, 1.1, 5, minSize=(30,30))  # [Dong 36] Quet tim mat: scaleFactor=1.1, minNeighbors=5, bo qua mat < 30x30px

    def predict(self, face_img):  # [Dong 38] Dinh nghia ham du doan cam xuc cho mot vung anh khuon mat da cat ra
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)  # [Dong 39] Chuyen anh cat khuon mat sang anh xam Grayscale de chuan bi can bang sang
        eq = self.clahe.apply(gray)  # [Dong 40] Ap dung CLAHE de can bang sang deu tren cac vung mat, lam ro net cac nep nhan va tho co bieu cam
        rgb = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)  # [Dong 41] Chuyen lai anh xam sang 3 kenh RGB vi model EfficientNet yeu cau dau vao 3 kenh mau
        name, scores = self.er.predict_emotions(rgb, logits=False)  # [Dong 42] Chay model predict tra ve mảng 8 xac suat cam xuc goc
        # Anh xa 8 lop cua HSEmotion sang 6 lop cua bai toan  # [Dong 43] Chu thich phep anh xa
        raw6 = np.array([scores[0], scores[2], scores[4], scores[5], scores[6], scores[7]])  # [Dong 44] Loc lay 6 lop, bo Fear (chi so 1) va Contempt (chi so 3)
        scores = raw6 / raw6.sum() if raw6.sum() > 0 else raw6  # [Dong 45] Tai chuan hoa Softmax rut gon: chia tung phan tu cho tong de tong xac suat = 100%
        return scores  # [Dong 46] Tra ve mang 6 gia tri xac suat da chuan hoa cua khuon mat

    def process(self, frame):  # [Dong 48] Dinh nghia ham tong xu ly luong tren 1 khung hinh (detect, tracking, predict, smoothing, threshold)
        results = []  # [Dong 49] Khoi tao list result trong de chua thong tin ket qua cac mat trong khung hinh nay
        faces = self.detect(frame)  # [Dong 50] Goi ham detect o dong 34 de phat hien tat ca cac khuon mat trong frame
        if len(faces) > 1:  # [Dong 51] Neu phat hien tu 2 khuon mat tro len
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)  # [Dong 52] Sap xep mat theo dien tich giam dan (f[2]*f[3]). Mat lon nhat o vi tri dau (index 0)
        active = set()  # [Dong 53] Khoi tao tap hop active de luu tru toa do tam cua cac khuon mat dang thuc su xuat hien tren frame
        for (x,y,w,h) in faces:  # [Dong 54] Vong lap xet qua tung khuon mat
            if w>=48 and h>=48:  # [Dong 55] Chi quan tam den cac khuon mat co do rong va dai tu 48px tro len
                k = ((x+w//2)//self.GRID*self.GRID, (y+h//2)//self.GRID*self.GRID)  # [Dong 56] Centroid Tracking: Luong tu hoa tam mat ve o luoi 60px lam khoa dinh danh k
                active.add(k)  # [Dong 57] Them khoa dinh danh k vao tap hop active
        for k in set(self.bufs.keys())-active:  # [Dong 58] Tim cac khuon mat co trong bo dem tu dien nhung khong con xuat hien tren camera
            del self.bufs[k]  # [Dong 59] Xoa bo dem cua khuon mat da roi di khoi camera de tranh bi ro ri bo nho RAM
        for (x,y,w,h) in faces:  # [Dong 60] Vong lap thu hai de thuc hien predict cho tung mat
            if w<48 or h<48: continue  # [Dong 61] Bỏ qua ngay cac khuon mặt o qua xa, qua nhỏ (nho hon 48x48 pixel)
            k = ((x+w//2)//self.GRID*self.GRID, (y+h//2)//self.GRID*self.GRID)  # [Dong 62] Tinh khoa dinh danh k de dinh vi dung nguoi trong bufs
            face = frame[y:y+h,x:x+w]  # [Dong 63] Cat lay rieng vung anh ma tran chua khuon mat (Crop) tu frame anh goc
            scores = self.predict(face)  # [Dong 64] Chay ham predict o dong 38 de lay mảng 6 xac suat cam xuc
            if k not in self.bufs: self.bufs[k] = deque(maxlen=self.BSZ)  # [Dong 65] Neu la nguoi moi, khoi tao hang doi deque luu toi da 10 khung hinh
            self.bufs[k].append(scores)  # [Dong 66] Day mang xac suat moi vao hang doi deque cua nguoi do
            avg = np.mean(self.bufs[k], axis=0)  # [Dong 67] Temporal Smoothing: Tinh trung binh cong xac suat cua 10 khung hinh gan nhat
            idx = np.argmax(avg)  # [Dong 68] Tim chi so index cua cam xuc dat xac suat trung binh lon nhat
            conf = float(avg[idx])  # [Dong 69] Lay gia tri xac suat cao nhat do (do tin cay) ep kieu Float
            emo = EMOTIONS[idx]  # [Dong 70] Tra cuu ten cam xuc tu mảng EMOTIONS dua vao chi so idx vua tim duoc
            if conf < self.THRESH.get(emo, .4):  # [Dong 71] Kiem tra nguong thich nghi: neu do tin cay thap hon nguong quy dinh (vi du Happiness < 0.3)
                emo, conf = 'Neutral', float(avg[EMOTIONS.index('Neutral')])  # [Dong 72] Ep cam xuc ve nhan mac dinh Neutral va lay dung xac suat lop Neutral tuong ung
            else:  # [Dong 73] Truong hop dat tren nguong thich nghi
                emo = DISPLAY.get(emo, emo)  # [Dong 74] Doi ten sang nhan hien thi than thien tren UI (vi du: Happiness -> Happy)
            results.append({'bbox':(x,y,w,h),'emotion':emo,'confidence':conf,'probs':avg})  # [Dong 75] Them ket qua gom bounding box, nhan, do tin cay vao list results
        return results  # [Dong 76] Tra ve danh sach kết quả phan tich cac khuon mat cua khung hinh nay

# ── Session Stats ──  # [Dong 78] Khai bao Class SessionStats - ghi nhan thong ke phien hoc va tu dong ve do thi Dashboard
class SessionStats:  # [Dong 79] Dinh nghia lop SessionStats
    def __init__(self):  # [Dong 80] Ham khoi tao bo thong ke
        self.history = []  # [Dong 81] Khoi tao danh sach history trong de ghi nhan lich su cam xuc theo thoi gian thuc tren RAM
        self.order = list(EMO_COLORS.keys())  # [Dong 82] Lay danh sach ten cam xuc theo ma mau de sap xep thu tu truc Y tren bieu do timeline

    def record(self, t, results):  # [Dong 84] Dinh nghia ham ghi nhan lich su tai thoi diem t (giay)
        if results:  # [Dong 85] Neu co ket qua phat hien mat tren frame
            r = results[0]  # [Dong 86] Chi lay khuon mat lon nhat ở vi tri results[0] (Primary Face - hoc sinh chính), bo qua nhiễu sau lung
            self.history.append({'time':t,'emotion':r['emotion'],'confidence':r['confidence']})  # [Dong 87] Append O(1) thoi gian, nhan, do tin cay vao RAM

    def report(self, out_dir='reports'):  # [Dong 89] Dinh nghia ham ve va luu Dashboard reports/emotion_report.png
        if not self.history: return  # [Dong 90] Neu danh sach lich su trong (chua ghi duoc gi), dung ham khong ve
        os.makedirs(out_dir, exist_ok=True)  # [Dong 91] Tao thu muc reports/ tren o cung neu thu muc nay chua ton tai
        ts = [h['time'] for h in self.history]  # [Dong 92] Tach rieng danh sach thoi gian ts tu lich su
        emos = [h['emotion'] for h in self.history]  # [Dong 93] Tach rieng danh sach cac nhan cam xuc emos tu lich su
        confs = [h['confidence'] for h in self.history]  # [Dong 94] Tach rieng danh sach cac do tin cay confs tu lich su
        cnt = Counter(emos)  # [Dong 95] Dem tan suat xuat hien tung cam xuc de ve bieu do tron
        present = [e for e in self.order if e in cnt]  # [Dong 96] Loc cac cam xuc thuc te co xuat hien trong phien hoc
        counts = [cnt[e] for e in present]  # [Dong 97] Lay so lan xuat hien tuong ung voi tung cam xuc do
        colors = [tuple(c/255 for c in reversed(EMO_COLORS[e])) for e in present]  # [Dong 98] Chuyen doi ma mau BGR sang ti le RGB (0.0 - 1.0) cua Matplotlib
        emo2y = {e:i for i,e in enumerate(self.order)}  # [Dong 99] Anh xa ten cam xuc sang index so nguyen 0-5 de lam toa do truc dung Y
        yv = [emo2y[e] for e in emos]  # [Dong 100] Tao mang yv chua index truc dung tuong ung cho toan bo chuoi cam xuc lich su

        fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12,10),gridspec_kw={'height_ratios':[1,1.5]})  # [Dong 102] Tao khung bieu do size 12x10 inch, chia lam 2 do thi chong doc: ax1 (tren, ti le 1.0), ax2 (duoi, ti le 1.5)
        ax1.pie(counts, labels=present, colors=colors, autopct=lambda p:f'{p:.1f}%', startangle=90)  # [Dong 103] Ve do thi tron (Pie Chart) phan bo thoi luong cam xuc. Goc bat dau 90 do
        ax1.set_title('Phan Bo Cam Xuc')  # [Dong 104] Dat ten tieu de cho bieu do tron la "Phan Bo Cam Xuc"
        ax2.step(ts, yv, where='post', color='gray', alpha=.5, linewidth=1.5, linestyle='--')  # [Dong 105] Ve bieu do buoc duong dut net xam noi cac moc cam xuc de thay su bien doi
        for t,e,c in zip(ts,emos,confs):  # [Dong 106] Duyet qua thoi gian, nhan, do tin cay de ve cac cham tron
            rgb = tuple(v/255 for v in reversed(EMO_COLORS[e]))  # [Dong 107] Lay ma mau RGB phu hop cho cham tron tuong ung voi cam xuc do
            ax2.scatter(t, emo2y[e], s=max(40,c*250), color=[rgb], edgecolors='black', linewidth=.5)  # [Dong 108] Ve cham tron, size s co gian dong theo do tin cay = max(40, c*250), co vien den
        ax2.set_yticks(range(len(self.order)))  # [Dong 109] Thiet lap so vach chia tren truc Y bang dung so cam xuc (0 den 5)
        ax2.set_yticklabels(self.order)  # [Dong 110] Gan chu ten cam xuc tuong ung vao cac vach chia truc Y
        ax2.set_xlabel('Thoi gian (giay)')  # [Dong 111] Dat ten nhan truc nam ngang X la "Thoi gian (giay)"
        ax2.set_ylabel('Cam xuc')  # [Dong 112] Dat ten nhan truc dung Y la "Cam xuc"
        ax2.grid(True, alpha=.3)  # [Dong 113] Bat luoi bieu do voi do trong suot 0.3 de de giong hang toa do
        plt.tight_layout()  # [Dong 114] Căn chinh tu dong de cac chu tren bieu do khong bi chong cheo hoac mat vien
        plt.savefig(os.path.join(out_dir,'emotion_report.png'), dpi=150, bbox_inches='tight')  # [Dong 115] Luu toan bo hinh anh Dashboard thanh file reports/emotion_report.png voi DPI = 150
        plt.close()  # [Dong 116] Dong doi tuong do thi Matplotlib de giai phong hoan toan bo nho RAM
        print(f'  Da luu: reports/emotion_report.png')  # [Dong 117] In thong bao da ghi file Dashboard ra man hinh console

# ── Draw Helpers ──  # [Dong 119] Dinh nghia cac ham tro giup do hoa ve khung bounding box va viet chu nhan len giao dien camera
def draw_boxes(frame, results):  # [Dong 120] Dinh nghia ham draw_boxes, nhan vao khung hinh va ket qua results
    out = frame.copy()  # [Dong 121] Copy ra ban sao anh out de ve, giu nguyen ma tran anh thô frame ban dau khoi bi thay doi
    for r in results:  # [Dong 122] Vong lap duyet qua ket qua tung mat hien co tren khung hinh
        x,y,w,h = r['bbox']; emo = r['emotion']; conf = r['confidence']  # [Dong 123] Lay toa do khung hop (x,y,w,h), ten cam xuc, va do tin cay
        color = EMO_COLORS.get(emo,(255,255,255))  # [Dong 124] Lay mau sac ve khung tuong ung voi cam xuc, mac dinh la trang neu khong co
        cv2.rectangle(out,(x,y),(x+w,y+h),color,2)  # [Dong 125] Ve khung hop chu nhat net day 2px bao quanh khuon mat hoc sinh
        lbl = f'{emo}: {conf:.0%}'  # [Dong 126] Tao chuoi van ban hien thi: cam xuc + do tin cay lam tron phan tram (vi du: Happy: 95%)
        (lw,lh),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,.65,2)  # [Dong 127] Tinh kich thuoc o chu de thiet ke nen cho chu vua van
        cv2.rectangle(out,(x,y-lh-10),(x+lw+6,y),color,-1)  # [Dong 128] Ve o hop chu nhat dac (do day -1) lam nen cho chu nhan ngay tren khung mat
        cv2.putText(out,lbl,(x+3,y-5),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),2)  # [Dong 129] Viet nhan chu mau den (0,0,0) len tren o hop nen net day 2px
    return out  # [Dong 130] Tra ve khung hinh da duoc ve bounding box va ghi chu hoan thien

# ── Modes ──  # [Dong 132] Khai bao cac ham dieu khien ung dung theo tung che do chay đầu vao
def mode_webcam(args):  # [Dong 133] Dinh nghia ham chay Webcam thoi gian thuc
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # [Dong 134] Mo webcam bang DirectShow (Windows) de camera khoi dong len ngay lap tuc
    if not cap.isOpened(): cap = cv2.VideoCapture(args.camera)  # [Dong 135] Thu lai bang ham mac dinh neu DirectShow khong tuong thich thiet bi
    if not cap.isOpened(): print('Ko mo duoc camera!'); return  # [Dong 136] Neu camera bi lỗi hoac khong tim thay, in loi va dung chuong trinh
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)  # [Dong 137] Cau hinh webcam chay o do phan giai chuan 640x480 pixel
    eng = EmotionEngine(args.model)  # [Dong 138] Khoi tao bo xu ly cam xuc engine chay model EfficientNet
    stats = SessionStats(); fc=0; skip=2; last=[]; start_t = time.time()  # [Dong 139] Khoi tao stats ghi chet, bo dem frame fc=0, skip=2 de nhay frame tiet kiem CPU, start_t ghi moc bat dau

    os.makedirs('data', exist_ok=True)  # [Dong 141] Tu dong tao thu muc data/ dung de luu cac file anh chup man hinh camera

    try:  # [Dong 143] Khoi block try an toan bat buoc chay cap camera
        while True:  # [Dong 144] Vong lap vo han doc webcam thoi gian thuc
            ret,frame = cap.read()  # [Dong 145] Doc mot khung hinh moi tu camera. ret báo doc thanh cong, frame chua anh
            if not ret: break  # [Dong 146] Neu webcam bi mat ket noi hoac khong doc duoc anh, tu dong thoat vong lap
            fc+=1  # [Dong 147] Tang bien dem khung hinh them 1 don vi
            if fc%skip==0: last=eng.process(frame); stats.record(time.time()-start_t,last)  # [Dong 148] Nhay khung hinh: chi xu ly AI o frame chan (2,4,6...) de do nong may, ghi nhan stats
            display = draw_boxes(frame, last)  # [Dong 149] Ve bounding box va nhan len anh thuc te de hien thi
            cv2.putText(display,f'Frame:{fc} Faces:{len(last)}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)  # [Dong 150] In thong so so frame va so mat o goc tren ben trai
            cv2.putText(display,'q:thoat s:chup',(10,display.shape[0]-12),cv2.FONT_HERSHEY_SIMPLEX,.45,(200,200,200),1)  # [Dong 151] In chu huong dan phim tat duoi đáy ben trai
            cv2.imshow('Emotion Detection',display)  # [Dong 152] Mo cua so giao dien camera hien thi ket qua nhận dien cam xuc cho nguoi hoc xem
            k=cv2.waitKey(1)&0xFF  # [Dong 153] Doc phim bam cua nguoi dung thoi gian cho la 1 mili giay
            if k in (ord('q'),27): break  # [Dong 154] Neu nguoi dung an phim q hoac phim ESC (ma 27), ngat vong lap de tat
            elif k==ord('s'): cv2.imwrite(f'data/{datetime.now():%Y%m%d_%H%M%S}.jpg',display)  # [Dong 155] Neu nguoi dung an nut s, chup anh giao dien luu vao thu muc data/
    finally:  # [Dong 156] Khong ke chuong trinh chay binh thuong hay bi loi, luon luon thuc thi giai phong camera o day
        cap.release(); cv2.destroyAllWindows()  # [Dong 157] Tat webcam camera va tat toan bo cac cua so OpenCV tren man hinh
        stats.report()  # [Dong 158] Goi ham report o dong 89 de tu dong ve va xuat ra Dashboard reports/emotion_report.png

def mode_video(args):  # [Dong 160] Dinh nghia ham chay nhan dien cam xuc tu file video offline
    if not args.input or not os.path.isfile(args.input): print('Khong tim thay video!'); return  # [Dong 161] Neu khong co video dau vao, in loi va thoat
    cap = cv2.VideoCapture(args.input)  # [Dong 162] Mo file video dau vao bang OpenCV
    eng = EmotionEngine(args.model)  # [Dong 163] Khoi tao bo engine xu ly
    stats = SessionStats(); fc=0; skip=2; last=[]; start_t = time.time()  # [Dong 164] Khoi tao cac thong so ghi nhan thong ke tuong tu webcam
    fps = cap.get(cv2.CAP_PROP_FPS) or 25  # [Dong 165] Lay toc do khung hinh (FPS) cua video de dong bo thoi gian ghi file dau ra
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # [Dong 166] Lay tong so khung hinh cua video dung de tinh toan thanh tien trinh (progress bar)
    writer = None  # [Dong 167] Khoi tao bien ghi video writer mac dinh bang None
    if args.output:  # [Dong 168] Neu nguoi dung truyen duong dan video dau ra muon luu ket qua
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # [Dong 169] Lay chieu rong va chieu cao dung cua video goc
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))  # [Dong 170] Khoi tao bo ghi VideoWriter MP4

    try:  # [Dong 172] Khoi block try an toan de doc video
        while True:  # [Dong 173] Vong lap vo han doc tung khung hinh cua video
            ret,frame = cap.read()  # [Dong 174] Doc frame tiep theo cua video
            if not ret: break  # [Dong 175] Khi video het khung hinh (het phim), tu dong thoat vong lap
            fc+=1  # [Dong 176] Tang bien dem khung hinh cua video them 1
            if fc%skip==0: last=eng.process(frame); stats.record(time.time()-start_t,last)  # [Dong 177] Nhay khung hinh: chi xử ly AI o frame chan de video chay muot, ghi nhan stats
            display = draw_boxes(frame, last)  # [Dong 178] Ve nhan cam xuc len frame hien tai
            prog = fc/total if total>0 else 0  # [Dong 179] Tinh phan tram tien do video hien tai tu 0.0 den 1.0
            cv2.rectangle(display,(0,display.shape[0]-6),(display.shape[1],display.shape[0]),(50,50,50),-1)  # [Dong 180] Ve o vuong xam dam o sat day man hinh lam duong ray cho thanh tien trinh
            cv2.rectangle(display,(0,display.shape[0]-6),(int(display.shape[1]*prog),display.shape[0]),(0,200,255),-1)  # [Dong 181] Ve de o mau vang dai bang dung prog x display.width de bieu thi thanh tien do chay
            cv2.putText(display,f'Frame {fc}/{total} Faces:{len(last)}',(10,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)  # [Dong 182] In thong tin tien trinh frame len goc trai man hinh
            if writer: writer.write(display)  # [Dong 183] Ghi ghi frame da ve nhan vao video dau ra neu co yeu cau ghi file
            cv2.imshow('Emotion Detection',display)  # [Dong 184] Hien thi cua so video len man hinh
            k=cv2.waitKey(1)&0xFF  # [Dong 185] Bat phim bam cua nguoi dung cho 1 mili giay
            if k in (ord('q'),27): break  # [Dong 186] Neu nhan q hoac ESC, dung video ngay lap tuc
            elif k==ord(' '):  # [Dong 187] Neu nhan nut SPACE (Dau cach)
                while True:  # [Dong 188] Chay vong lap vo han de dung phim (Pause video)
                    if cv2.waitKey(100)&0xFF!=ord(' '): continue  # [Dong 189] Neu chua bam lai nut SPACE, tiep tuc treo
                    else: break  # [Dong 190] Neu bam lai nut SPACE, tiep tuc chay tiep video
    finally:  # [Dong 191] Khai bao khoi finally dong va giai phong file video
        cap.release()  # [Dong 192] Ngat ket noi doc video
        if writer: writer.release()  # [Dong 193] Giai phong file video dau ra de ghi hoan tat du lieu xuat ra dia
        cv2.destroyAllWindows()  # [Dong 194] Dong cua so video tren man hinh
        stats.report()  # [Dong 195] Goi ve Dashboard emotion_report.png báo cao cuoi video

def mode_image(args):  # [Dong 197] Dinh nghia ham nhan dien cam xuc tren file anh tinh
    if not args.input or not os.path.isfile(args.input): print('Khong tim thay anh!'); return  # [Dong 198] Neu khong tim thay anh dau vao, in loi va thoat
    frame = cv2.imread(args.input)  # [Dong 199] Doc file anh vao bo nho bang OpenCV
    if frame is None: print('Ko doc duoc anh!'); return  # [Dong 200] Neu anh loi khong the doc duoc, in loi va thoat
    eng = EmotionEngine(args.model)  # [Dong 201] Khoi tao bo engine xu ly
    results = eng.process(frame)  # [Dong 202] Quet phat hien va nhan dạng cam xuc tat ca cac mat co trong anh
    display = draw_boxes(frame, results)  # [Dong 203] Ve bounding box va viet nhan len anh
    if args.output: cv2.imwrite(args.output, display); print(f'Da luu: {args.output}')  # [Dong 204] Neu co tham so output, luu anh ket qua xuong o cung
    cv2.imshow('Emotion Detection',display)  # [Dong 205] Mo cua so hien thi anh kết qua len man hinh
    cv2.waitKey(0); cv2.destroyAllWindows()  # [Dong 206] Treo cua so vo han cho den khi nguoi dung an 1 phim bat ky thi tat o

# ── Main ──  # [Dong 208] Vung dinh nghia diem khoi chay chuong trinh
if __name__ == '__main__':  # [Dong 209] Diem vao chinh thuc cua chuong trinh Python chay tu terminal
    p = argparse.ArgumentParser(description='Nhan dang cam xuc')  # [Dong 210] Khoi tao bo nhan tham so dong lenh argparse
    p.add_argument('--mode', default='webcam', choices=['webcam','image','video'])  # [Dong 211] Tham so --mode de chon che do chay: webcam (mac dinh), image, video
    p.add_argument('--model', default='enet_b0_8_best_afew')  # [Dong 212] Tham so --model de chon model cua HSEmotion. Mac dinh dung EfficientNet-B0
    p.add_argument('--camera', type=int, default=0)  # [Dong 213] Tham so --camera de chon cong camera ket noi (mac dinh la 0 - webcam goc)
    p.add_argument('--input', default=None)  # [Dong 214] Tham so --input nhan duong dan file video hoac anh dau vao
    p.add_argument('--output', default=None)  # [Dong 215] Tham so --output nhan duong dan file ghi video hoac ghi anh ket qua
    args = p.parse_args()  # [Dong 216] Phich va nap cac tham so người dung da nhap vao bien doi tuong args
    {'webcam':mode_webcam,'image':mode_image,'video':mode_video}[args.mode](args)  # [Dong 217] Ky thuat Dictionary Mapping goi dong ham (mode_webcam/mode_image/mode_video) tuong ung, truyen tham so args
