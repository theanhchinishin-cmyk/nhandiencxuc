"""Nhan dang emotion dung HSEmotion"""

import cv2
import os
import numpy as np
import torch
from hsemotion.facial_emotions import HSEmotionRecognizer
from collections import deque
from typing import Tuple, List, Optional, Dict
from src.constants import EMOTION_LABELS_8, EMOTION_DISPLAY_MAP


class EmotionEngine:
    """Nhan dang emotion nang cao su dung HSEmotion (EfficientNet)

    Tich hop:
    - Temporal Smoothing: lam min ket qua qua time (trung binh 10 frame)
    - CLAHE Preprocessing: can bang anh sang
    - Adaptive Thresholding: nguong rieng cho tung emotion
    """

    def __init__(self, model_name: str = 'enet_b0_8_best_afew',
                 weights_path: str = None):
        self.model_name = model_name
        self.emotion_recognizer = HSEmotionRecognizer(model_name=model_name)
        self.face_cascade = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        self.face_classifier = cv2.CascadeClassifier(self.face_cascade)
        if self.face_classifier.empty():
            print(" Ko load dc Haar Cascade!")
            print(f"  Da tim: {self.face_cascade}")

        # labels emotion (theo output model 8 classes)
        self.emotion_labels = EMOTION_LABELS_8[:]

        # load fine-tuned weights neu co
        if weights_path is not None:
            self._load_finetuned_weights(weights_path)

        # 1. Temporal Smoothing: moi face co buffer rieng
        self.face_buffers: Dict[tuple, deque] = {}
        self.BUFFER_SIZE = 10   # luu 10 frame gan nhat
        self.GRID_SIZE   = 60   # chiu duoc rung nho < 60px

        # 2. Adaptive Thresholds: nguong rieng cho tung emotion
        self.thresholds = {
            'Happiness': 0.50,
            'Sadness': 0.30,
            'Surprise': 0.35,
            'Anger': 0.40,
            'Neutral': 0.35
        }

        # 3. CLAHE: can bang anh sang
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        mode = f"Fine-tuned ({weights_path})" if weights_path else "Pretrained goc"
        print(f" EmotionEngine san sang | Model: {model_name} | Weights: {mode}")

    def _load_finetuned_weights(self, weights_path: str):
        """load fine-tuned weights vao model

        Luu y: HSEmotionRecognizer thay model.classifier = nn.Identity va
        luu classifier_weights/bias dang numpy rieng.
        Can update CA model state_dict + numpy classifier.
        """
        import os
        if not os.path.isfile(weights_path):
            print(f" Khong tim thay file weights: {weights_path}")
            print("   Tiep tuc voi model goc (pretrained).")
            return

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

        # 1. load backbone weights vao model
        self.emotion_recognizer.model.load_state_dict(
            checkpoint['model_state_dict'], strict=False
        )
        self.emotion_recognizer.model.to(device)

        # 2. update numpy classifier cua HSEmotionRecognizer
        sd = checkpoint['model_state_dict']
        if 'classifier.weight' in sd:
            ft_w = sd['classifier.weight'].cpu().numpy()
            ft_b = sd['classifier.bias'].cpu().numpy()
            self.emotion_recognizer.classifier_weights = ft_w
            self.emotion_recognizer.classifier_bias   = ft_b
            print(f"  Da update classifier head (numpy, {ft_w.shape})")

        # 3. dong bo device
        self.emotion_recognizer.device = device

        val_acc = checkpoint.get('val_acc', 0)
        epoch = checkpoint.get('epoch', '?')
        print(f" Da load fine-tuned weights: {weights_path}")
        print(f"  (Epoch {epoch} | Val accuracy: {val_acc:.1f}%)")

    def _preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """preprocess: gray -> CLAHE -> RGB"""
        if face_img.size == 0:
            return face_img

        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        equalized = self.clahe.apply(gray)
        processed = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
        return processed

    def _get_face_key(self, x: int, y: int, w: int, h: int) -> tuple:
        """tao key cho face dua tren tam, luong tu hoa vao luoi chiu rung"""
        cx = ((x + w // 2) // self.GRID_SIZE) * self.GRID_SIZE
        cy = ((y + h // 2) // self.GRID_SIZE) * self.GRID_SIZE
        return (cx, cy)

    def predict_emotion(self, face_img: np.ndarray,
                        face_key: tuple = (0, 0)) -> Tuple[str, float, np.ndarray]:
        """nhan dien emotion voi temporal smoothing + adaptive threshold

        Args:
            face_img: anh mat da crop
            face_key: key de lay buffer rieng cho tung face

        Returns:
            (emotion, confidence, prob_vector)
        """
        try:
            processed_face = self._preprocess_face(face_img)
            emotion_name_raw, scores = self.emotion_recognizer.predict_emotions(
                processed_face, logits=False
            )

            if face_key not in self.face_buffers:
                self.face_buffers[face_key] = deque(maxlen=self.BUFFER_SIZE)
            buf = self.face_buffers[face_key]
            buf.append(scores)

            avg_scores = np.mean(buf, axis=0)
            max_idx = np.argmax(avg_scores)
            max_conf = float(avg_scores[max_idx])
            emotion_candidate = self.emotion_labels[max_idx]

            threshold = self.thresholds.get(emotion_candidate, 0.40)

            if max_conf < threshold:
                final_emotion = 'Neutral'
                neutral_idx = self.emotion_labels.index('Neutral')
                confidence = float(avg_scores[neutral_idx])
            else:
                final_emotion = EMOTION_DISPLAY_MAP.get(emotion_candidate, emotion_candidate)
                confidence = max_conf

            return final_emotion, confidence, avg_scores

        except Exception as e:
            print(f" Loi predict emotion: {e}")
            return "Neutral", 0.0, np.zeros(len(self.emotion_labels))

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """detect mat bang Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """xu ly 1 frame: detect face + nhan dien emotion

        moi face duoc track bang face_key rieng
        """
        results = []
        faces = self.detect_faces(frame)

        # xoa buffer cua face ko con xuat hien
        active_keys = set()
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            if face_img.shape[0] >= 48 and face_img.shape[1] >= 48:
                active_keys.add(self._get_face_key(x, y, w, h))
        stale_keys = set(self.face_buffers.keys()) - active_keys
        for k in stale_keys:
            del self.face_buffers[k]

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            if face_img.shape[0] < 48 or face_img.shape[1] < 48:
                continue

            face_key = self._get_face_key(x, y, w, h)
            emotion, confidence, probs = self.predict_emotion(face_img, face_key)

            results.append({
                'bbox': (x, y, w, h),
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': probs
            })

        return results
