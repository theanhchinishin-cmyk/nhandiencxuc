"""
main.py - He thong nhan dang cam xuc nguoi hoc online
Su dung: python main.py [--mode webcam|image|video]
"""

import os, sys, time, argparse, numpy as np, cv2, torch
from collections import deque, Counter
from datetime import datetime
from hsemotion.facial_emotions import HSEmotionRecognizer
import matplotlib.pyplot as plt

# ── Constants ──
EMOTIONS = ['Anger','Disgust','Happiness','Neutral','Sadness','Surprise']
EMO_COLORS = {'Happy':(0,255,0),'Sad':(255,80,80),'Angry':(0,0,255),
              'Surprise':(0,165,255),'Neutral':(180,180,180),'Disgust':(0,160,160)}
DISPLAY = {'Anger':'Angry','Happiness':'Happy','Sadness':'Sad','Contempt':'Disgust'}
FACE_CASCADE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# ── Emotion Engine ──
class EmotionEngine:
    def __init__(self, model_name='enet_b0_8_best_afew'):
        # Override torch.load to bypass weights_only warning in HSEmotionRecognizer
        torch_load = torch.load
        torch.load = lambda f,**kw: torch_load(f,**{**kw,'weights_only':False})
        self.er = HSEmotionRecognizer(model_name=model_name)
        torch.load = torch_load
        
        self.fc = cv2.CascadeClassifier(FACE_CASCADE)
        self.bufs = {}
        self.BSZ, self.GRID = 10, 60
        self.THRESH = {'Happiness':.3,'Sadness':.2,'Surprise':.3,'Anger':.3,'Neutral':.35,'Disgust':.3}
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.fc.detectMultiScale(gray, 1.1, 5, minSize=(30,30))

    def predict(self, face_img):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        eq = self.clahe.apply(gray)
        rgb = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
        name, scores = self.er.predict_emotions(rgb, logits=False)
        # Anh xa 8 lop cua HSEmotion sang 6 lop cua bai toan
        raw6 = np.array([scores[0], scores[2], scores[4], scores[5], scores[6], scores[7]])
        scores = raw6 / raw6.sum() if raw6.sum() > 0 else raw6
        return scores

    def process(self, frame):
        results = []
        faces = self.detect(frame)
        if len(faces) > 1:
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        active = set()
        for (x,y,w,h) in faces:
            if w>=48 and h>=48:
                k = ((x+w//2)//self.GRID*self.GRID, (y+h//2)//self.GRID*self.GRID)
                active.add(k)
        for k in set(self.bufs.keys())-active:
            del self.bufs[k]
        for (x,y,w,h) in faces:
            if w<48 or h<48: continue
            k = ((x+w//2)//self.GRID*self.GRID, (y+h//2)//self.GRID*self.GRID)
            face = frame[y:y+h,x:x+w]
            scores = self.predict(face)
            if k not in self.bufs: self.bufs[k] = deque(maxlen=self.BSZ)
            self.bufs[k].append(scores)
            avg = np.mean(self.bufs[k], axis=0)
            idx = np.argmax(avg)
            conf = float(avg[idx])
            emo = EMOTIONS[idx]
            if conf < self.THRESH.get(emo, .4):
                emo, conf = 'Neutral', float(avg[EMOTIONS.index('Neutral')])
            else:
                emo = DISPLAY.get(emo, emo)
            results.append({'bbox':(x,y,w,h),'emotion':emo,'confidence':conf,'probs':avg})
        return results

# ── Session Stats ──
class SessionStats:
    def __init__(self):
        self.history = []
        self.order = list(EMO_COLORS.keys())

    def record(self, t, results):
        if results:
            r = results[0]
            self.history.append({'time':t,'emotion':r['emotion'],'confidence':r['confidence']})

    def report(self, out_dir='reports'):
        if not self.history: return
        os.makedirs(out_dir, exist_ok=True)
        ts = [h['time'] for h in self.history]
        emos = [h['emotion'] for h in self.history]
        confs = [h['confidence'] for h in self.history]
        cnt = Counter(emos)
        present = [e for e in self.order if e in cnt]
        counts = [cnt[e] for e in present]
        colors = [tuple(c/255 for c in reversed(EMO_COLORS[e])) for e in present]
        emo2y = {e:i for i,e in enumerate(self.order)}
        yv = [emo2y[e] for e in emos]

        fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12,10),gridspec_kw={'height_ratios':[1,1.5]})
        ax1.pie(counts, labels=present, colors=colors, autopct=lambda p:f'{p:.1f}%', startangle=90)
        ax1.set_title('Phan Bo Cam Xuc')
        ax2.step(ts, yv, where='post', color='gray', alpha=.5, linewidth=1.5, linestyle='--')
        for t,e,c in zip(ts,emos,confs):
            rgb = tuple(v/255 for v in reversed(EMO_COLORS[e]))
            ax2.scatter(t, emo2y[e], s=max(40,c*250), color=[rgb], edgecolors='black', linewidth=.5)
        ax2.set_yticks(range(len(self.order)))
        ax2.set_yticklabels(self.order)
        ax2.set_xlabel('Thoi gian (giay)')
        ax2.set_ylabel('Cam xuc')
        ax2.grid(True, alpha=.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,'emotion_report.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Da luu: reports/emotion_report.png')

# ── Draw Helpers ──
def draw_boxes(frame, results):
    out = frame.copy()
    for r in results:
        x,y,w,h = r['bbox']; emo = r['emotion']; conf = r['confidence']
        color = EMO_COLORS.get(emo,(255,255,255))
        cv2.rectangle(out,(x,y),(x+w,y+h),color,2)
        lbl = f'{emo}: {conf:.0%}'
        (lw,lh),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,.65,2)
        cv2.rectangle(out,(x,y-lh-10),(x+lw+6,y),color,-1)
        cv2.putText(out,lbl,(x+3,y-5),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),2)
    return out

# ── Modes ──
def mode_webcam(args):
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened(): print('Ko mo duoc camera!'); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    eng = EmotionEngine(args.model)
    stats = SessionStats(); fc=0; skip=2; last=[]; start_t = time.time()

    os.makedirs('data', exist_ok=True)

    try:
        while True:
            ret,frame = cap.read()
            if not ret: break
            fc+=1
            if fc%skip==0: last=eng.process(frame); stats.record(time.time()-start_t,last)
            display = draw_boxes(frame, last)
            cv2.putText(display,f'Frame:{fc} Faces:{len(last)}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
            cv2.putText(display,'q:thoat s:chup',(10,display.shape[0]-12),cv2.FONT_HERSHEY_SIMPLEX,.45,(200,200,200),1)
            cv2.imshow('Emotion Detection',display)
            k=cv2.waitKey(1)&0xFF
            if k in (ord('q'),27): break
            elif k==ord('s'): cv2.imwrite(f'data/{datetime.now():%Y%m%d_%H%M%S}.jpg',display)
    finally:
        cap.release(); cv2.destroyAllWindows()
        stats.report()

def mode_video(args):
    if not args.input or not os.path.isfile(args.input): print('Khong tim thay video!'); return
    cap = cv2.VideoCapture(args.input)
    eng = EmotionEngine(args.model)
    stats = SessionStats(); fc=0; skip=2; last=[]; start_t = time.time()
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None
    if args.output:
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

    try:
        while True:
            ret,frame = cap.read()
            if not ret: break
            fc+=1
            if fc%skip==0: last=eng.process(frame); stats.record(time.time()-start_t,last)
            display = draw_boxes(frame, last)
            prog = fc/total if total>0 else 0
            cv2.rectangle(display,(0,display.shape[0]-6),(display.shape[1],display.shape[0]),(50,50,50),-1)
            cv2.rectangle(display,(0,display.shape[0]-6),(int(display.shape[1]*prog),display.shape[0]),(0,200,255),-1)
            cv2.putText(display,f'Frame {fc}/{total} Faces:{len(last)}',(10,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
            if writer: writer.write(display)
            cv2.imshow('Emotion Detection',display)
            k=cv2.waitKey(1)&0xFF
            if k in (ord('q'),27): break
            elif k==ord(' '):
                while True:
                    if cv2.waitKey(100)&0xFF!=ord(' '): continue
                    else: break
    finally:
        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()
        stats.report()

def mode_image(args):
    if not args.input or not os.path.isfile(args.input): print('Khong tim thay anh!'); return
    frame = cv2.imread(args.input)
    if frame is None: print('Ko doc duoc anh!'); return
    eng = EmotionEngine(args.model)
    results = eng.process(frame)
    display = draw_boxes(frame, results)
    if args.output: cv2.imwrite(args.output, display); print(f'Da luu: {args.output}')
    cv2.imshow('Emotion Detection',display)
    cv2.waitKey(0); cv2.destroyAllWindows()

# ── Main ──
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Nhan dang cam xuc')
    p.add_argument('--mode', default='webcam', choices=['webcam','image','video'])
    p.add_argument('--model', default='enet_b0_8_best_afew')
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--input', default=None)
    p.add_argument('--output', default=None)
    args = p.parse_args()
    {'webcam':mode_webcam,'image':mode_image,'video':mode_video}[args.mode](args)
