"""
main.py - He thong nhan dang cam xuc
Su dung: python main.py [--mode webcam|image|video|finetune|evaluate]
"""

import os, sys, time, argparse, numpy as np, cv2, torch, torch.nn as nn
import torch.optim as optim, matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms, datasets
from collections import deque, Counter
from datetime import datetime
import timm
from hsemotion.facial_emotions import HSEmotionRecognizer, get_model_path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# ── Constants ──
EMOTIONS = ['Anger','Disgust','Happiness','Neutral','Sadness','Surprise']
EMO_COLORS = {'Happy':(0,255,0),'Sad':(255,80,80),'Angry':(0,0,255),
              'Surprise':(0,165,255),'Neutral':(180,180,180),'Disgust':(0,160,160)}
DISPLAY = {'Anger':'Angry','Happiness':'Happy','Sadness':'Sad','Contempt':'Disgust'}
FACE_CASCADE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
SUPPORTED = ('.jpg','.jpeg','.png','.webp')

# ── Emotion Engine ──
class EmotionEngine:
    def __init__(self, model_name='enet_b0_8_best_afew', weights_path=None):
        torch_load = torch.load
        torch.load = lambda f,**kw: torch_load(f,**{**kw,'weights_only':False})
        self.er = HSEmotionRecognizer(model_name=model_name)
        torch.load = torch_load
        self.fc = cv2.CascadeClassifier(FACE_CASCADE)
        self.is_ft = weights_path is not None
        if weights_path: self._load_weights(weights_path)
        self.bufs = {}
        self.BSZ, self.GRID = 10, 60
        self.THRESH = {'Happiness':.5,'Sadness':.3,'Surprise':.35,'Anger':.4,'Neutral':.35}
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def _load_weights(self, path):
        ckpt = torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
        self.er.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        sd = ckpt['model_state_dict']
        if 'classifier.weight' in sd:
            self.er.classifier_weights = sd['classifier.weight'].cpu().numpy()
            self.er.classifier_bias = sd['classifier.bias'].cpu().numpy()

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.fc.detectMultiScale(gray, 1.1, 5, minSize=(30,30))

    def predict(self, face_img):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        eq = self.clahe.apply(gray)
        rgb = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
        name, scores = self.er.predict_emotions(rgb, logits=False)
        if not self.is_ft:
            raw6 = np.array([scores[0], scores[2], scores[4], scores[5], scores[6], scores[7]])
            scores = raw6 / raw6.sum() if raw6.sum()>0 else raw6
        else:
            scores = scores[:6]
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

# ── Model Loader (for finetune) ──
def load_model(model_name='enet_b0_8_best_afew', num_classes=8):
    path = get_model_path(model_name)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = ckpt.state_dict() if hasattr(ckpt,'state_dict') else ckpt
    if 'classifier.0.weight' in sd:
        sd['classifier.weight'] = sd.pop('classifier.0.weight')
        sd['classifier.bias'] = sd.pop('classifier.0.bias')
    arch = 'tf_efficientnet_b2.ns_jft_in1k' if 'b2' in model_name else 'tf_efficientnet_b0.ns_jft_in1k'
    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    # Chi load classifier neu shape khop (pretrained=8, fine-tune=6)
    if 'classifier.weight' in sd and sd['classifier.weight'].shape[0] != num_classes:
        del sd['classifier.weight']
        del sd['classifier.bias']
    model.load_state_dict(sd, strict=False)
    return model

# ── Finetune ──
def do_finetune(dataset_dir, epochs=15, lr=1e-4, batch_size=16, save_path='models/finetuned.pt', model_name='enet_b0_8_best_afew'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Thiet bi: {device.upper()}')
    model = load_model(model_name, 6).to(device)
    for p in model.parameters(): p.requires_grad = False
    for block in list(model.blocks)[-2:]:
        for p in block.parameters(): p.requires_grad = True
    for p in model.classifier.parameters(): p.requires_grad = True

    tfmt = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225])])

    raw = datasets.ImageFolder(dataset_dir, is_valid_file=lambda p: p.lower().endswith(SUPPORTED))
    label_map = {}
    class_names = []
    for i,cls in enumerate(raw.classes):
        if cls in EMOTIONS:
            label_map[i] = EMOTIONS.index(cls)
            class_names.append(cls)
    if not class_names: raise ValueError('Khong co class hop le!')
    full = datasets.ImageFolder(dataset_dir, transform=tfmt,
        target_transform=lambda y: label_map.get(y,0), is_valid_file=lambda p: p.lower().endswith(SUPPORTED))

    n = len(full)
    n_val = max(1, int(n*.15))
    n_test = max(1, int(n*.15))
    n_train = n - n_val - n_test
    if n_train<=0: raise ValueError('Dataset qua nho!')
    train_ds,val_ds,test_ds = random_split(full, [n_train,n_val,n_test], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False)
    print(f'Dataset: {n} anh | Train: {n_train} Val: {n_val} Test: {n_test}')
    print(f'Class: {class_names}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p:p.requires_grad,model.parameters()), lr=lr)
    best_acc, best_hist = 0, {'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[]}

    for epoch in range(epochs):
        model.train()
        tl=tc=tt=0
        for imgs,labels in train_loader:
            imgs,labels = imgs.to(device),labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward(); optimizer.step()
            tl+=loss.item()
            _,pred = model(imgs).max(1)
            tc+=pred.eq(labels).sum().item(); tt+=labels.size(0)
        model.eval()
        vl=vc=vt=0
        with torch.no_grad():
            for imgs,labels in val_loader:
                imgs,labels = imgs.to(device),labels.to(device)
                out = model(imgs); loss = criterion(out,labels)
                vl+=loss.item()
                _,pred = out.max(1)
                vc+=pred.eq(labels).sum().item(); vt+=labels.size(0)
        ta,va = 100*tc/tt, 100*vc/vt
        best_hist['train_loss'].append(tl/len(train_loader))
        best_hist['val_loss'].append(vl/len(val_loader))
        best_hist['train_acc'].append(ta)
        best_hist['val_acc'].append(va)
        saved=''
        if va>best_acc:
            best_acc=va
            torch.save({'model_state_dict':model.state_dict(),'class_names':class_names,
                'model_name':model_name,'val_acc':va,'epoch':epoch+1,
                'train_history': best_hist}, save_path)
            saved=' *'
        print(f'Epoch {epoch+1:2d}/{epochs} TL: {tl/len(train_loader):.4f} TA: {ta:.1f}% VL: {vl/len(val_loader):.4f} VA: {va:.1f}%{saved}')
    print(f'\nBest val acc: {best_acc:.1f}% | Saved: {save_path}')
    return model, test_loader, class_names, best_hist

# ── Evaluate ──
def do_evaluate(model_name, weights_path, test_loader, class_names, history):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base = load_model(model_name, 8).to(device)
    ft = load_model(model_name, 6).to(device)
    ft.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False)['model_state_dict'], strict=False)
    base.eval(); ft.eval()
    def run(m, dl, orig):
        all_y, all_p = [], []
        with torch.no_grad():
            for imgs,labels in dl:
                imgs = imgs.to(device)
                out = m(imgs)
                if orig: out = out[:,[0,2,4,5,6,7]]
                else: out = out[:,:6]
                _,p = out.max(1)
                all_y.extend(labels.numpy()); all_p.extend(p.cpu().numpy())
        return np.array(all_y), np.array(all_p)
    yt, yb = run(base, test_loader, True)
    _, yf = run(ft, test_loader, False)
    ab, af = 100*accuracy_score(yt,yb), 100*accuracy_score(yt,yf)
    print(f'Base: {ab:.1f}% Fine-tuned: {af:.1f}% ({af-ab:+.1f}%)')
    pi = sorted(set(yt.tolist()))
    pn = [EMOTIONS[i] for i in pi if i<len(EMOTIONS)]
    print(classification_report(yt, yf, labels=pi, target_names=pn, zero_division=0))
    cm = confusion_matrix(yt, yf, labels=pi)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=pn, yticklabels=pn)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png', dpi=150); plt.close()
    print('Da luu: reports/confusion_matrix.png')
    if history:
        epochs = range(1,len(history['train_loss'])+1)
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))
        ax1.plot(epochs, history['train_loss'],'b-o',ms=4,label='Train')
        ax1.plot(epochs,history['val_loss'],'r-o',ms=4,label='Val')
        ax1.set_title('Loss'); ax1.legend(); ax1.grid(True,alpha=.3)
        ax2.plot(epochs,history['train_acc'],'b-o',ms=4,label='Train')
        ax2.plot(epochs,history['val_acc'],'r-o',ms=4,label='Val')
        ax2.set_title('Accuracy (%)'); ax2.legend(); ax2.grid(True,alpha=.3)
        plt.tight_layout()
        plt.savefig('reports/training_curve.png', dpi=150); plt.close()
        print('Da luu: reports/training_curve.png')

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
    eng = EmotionEngine(args.model, args.weights)
    stats = SessionStats(); fc=0; skip=2; last=[]; start_t = time.time()

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
    eng = EmotionEngine(args.model, args.weights)
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
    eng = EmotionEngine(args.model, args.weights)
    results = eng.process(frame)
    display = draw_boxes(frame, results)
    if args.output: cv2.imwrite(args.output, display); print(f'Da luu: {args.output}')
    cv2.imshow('Emotion Detection',display)
    cv2.waitKey(0); cv2.destroyAllWindows()

def mode_finetune(args):
    if not args.dataset: print('Can --dataset!'); return
    model, test_loader, class_names, history = do_finetune(
        args.dataset, args.epochs, args.lr, args.batch_size, args.save_weights, args.model)
    print('\nFinetune xong! Chay evaluate de danh gia.')

def mode_evaluate(args):
    if not args.dataset: print('Can --dataset!'); return
    weights = args.weights or args.save_weights
    if not os.path.isfile(weights): print(f'Khong tim thay weights: {weights}'); return
    # Load dataset (khong train)
    tfmt = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225])])
    raw = datasets.ImageFolder(args.dataset, is_valid_file=lambda p: p.lower().endswith(SUPPORTED))
    label_map = {}
    for i,cls in enumerate(raw.classes):
        if cls in EMOTIONS: label_map[i] = EMOTIONS.index(cls)
    full = datasets.ImageFolder(args.dataset, transform=tfmt,
        target_transform=lambda y: label_map.get(y,0),
        is_valid_file=lambda p: p.lower().endswith(SUPPORTED))
    if "test" in args.dataset.lower() or "dung1landuynhat" in args.dataset.lower():
        test_loader = DataLoader(full, args.batch_size, shuffle=False)
    else:
        n = len(full); n_val = max(1,int(n*.15)); n_test = max(1,int(n*.15))
        n_train = n - n_val - n_test
        _,_,test_ds = random_split(full, [n_train,n_val,n_test], generator=torch.Generator().manual_seed(42))
        test_loader = DataLoader(test_ds, args.batch_size, shuffle=False)
    class_names = [c for c in raw.classes if c in EMOTIONS]
    history = torch.load(weights, map_location='cpu', weights_only=False).get('train_history')
    do_evaluate(args.model, weights, test_loader, class_names, history)

# ── Main ──
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Nhan dang cam xuc')
    p.add_argument('--mode', default='webcam', choices=['webcam','image','video','finetune','evaluate'])
    p.add_argument('--model', default='enet_b0_8_best_afew')
    p.add_argument('--weights', default=None)
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--input', default=None)
    p.add_argument('--output', default=None)
    p.add_argument('--dataset', default=None)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=16, dest='batch_size')
    p.add_argument('--save-weights', default='models/finetuned.pt', dest='save_weights')
    args = p.parse_args()
    {'webcam':mode_webcam,'image':mode_image,'video':mode_video,
     'finetune':mode_finetune,'evaluate':mode_evaluate}[args.mode](args)
