"""Nhan dien emotion tu anh tinh"""

import cv2
import numpy as np
import os
from src.emotion_engine import EmotionEngine
from src.constants import EMOTION_COLORS, EMOTION_DISPLAY_MAP


class ImagePredictor:
    """Nhan dien emotion tu file anh (.jpg, .png, ...)
    Ho tro ca model goc va fine-tuned
    """

    COLOR_MAP = EMOTION_COLORS.copy()

    def __init__(self, model_name: str = 'enet_b0_8_best_afew',
                 weights_path: str = None):
        self.engine = EmotionEngine(
            model_name=model_name,
            weights_path=weights_path
        )

    def predict(self, image_path: str,
                output_path: str = None,
                show: bool = True) -> list:
        """nhan dien emotion trong anh

        Args:
            image_path: duong dan anh
            output_path: luu anh ket qua (None = ko luu)
            show: hien thi cua so (True/False)

        Returns:
            list ket qua [{emotion, confidence, bbox, probabilities}]
        """
        if not os.path.isfile(image_path):
            print(f" Ko tim thay anh: {image_path}")
            return []

        frame = cv2.imread(image_path)
        if frame is None:
            print(f" Ko doc duoc anh: {image_path}")
            return []

        results = self.engine.process_frame(frame)

        if not results:
            print(" Ko phat hien khuon mat trong anh.")
            if show:
                cv2.putText(frame, "No face detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            print(f" Phat hien {len(results)} khuon mat:")
            for i, result in enumerate(results):
                frame = self._draw_result(frame, result, i)
                emotion = result['emotion']
                conf = result['confidence']
                print(f"  Khuon mat #{i+1}: {emotion} ({conf:.1%})")

        if results:
            frame = self._draw_prob_bars(frame, results[0])

        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            cv2.imwrite(output_path, frame)
            print(f" Da luu anh ket qua: {output_path}")

        if show:
            window_name = "Emotion Detection - Image Mode"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            h, w = frame.shape[:2]
            cv2.resizeWindow(window_name, min(w, 1024), min(h, 720))
            cv2.imshow(window_name, frame)
            print(" Nhan phim bat ky de thoat...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return results

    # ──────────────────────────────────────────────
    # drawing helpers
    # ──────────────────────────────────────────────

    def _draw_result(self, frame: np.ndarray, result: dict, idx: int) -> np.ndarray:
        """ve bounding box + label len frame"""
        x, y, w, h = result['bbox']
        emotion = result['emotion']
        confidence = result['confidence']
        color = self.COLOR_MAP.get(emotion, (255, 255, 255))

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        label = f"{emotion}  {confidence:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y - lh - 12), (x + lw + 6, y), color, -1)
        cv2.putText(frame, label, (x + 3, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        return frame

    def _draw_prob_bars(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """ve thanh xac suat emotion o goc phai"""
        probs = result['probabilities']
        labels = self.engine.emotion_labels
        display_map = EMOTION_DISPLAY_MAP.copy()

        h, w = frame.shape[:2]
        bar_x = w - 220
        bar_max_w = 190
        bar_h = 22
        padding = 6
        n_bars = len(labels)
        total_height = 10 + n_bars * (bar_h + padding) + 10

        if total_height > h:
            return frame

        overlay = frame.copy()
        cv2.rectangle(overlay, (bar_x - 8, 10), (w - 5, total_height), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        sorted_idx = np.argsort(probs)[::-1]

        # gop duplicate display name (vd Contempt + Disgust -> Disgust)
        seen_display_names = {}
        for orig_i in sorted_idx:
            name = display_map.get(labels[orig_i], labels[orig_i])
            prob = float(probs[orig_i])
            if name in seen_display_names:
                seen_display_names[name] += prob
            else:
                seen_display_names[name] = prob

        for rank, (name, prob) in enumerate(
            sorted(seen_display_names.items(), key=lambda x: x[1], reverse=True)
        ):
            y_pos = 15 + rank * (bar_h + padding)
            bar_color = self.COLOR_MAP.get(name, (100, 180, 100))
            if rank == 0:
                bar_color = tuple(min(255, c + 60) for c in bar_color)

            bar_len = int(prob * bar_max_w)
            cv2.rectangle(frame, (bar_x, y_pos),
                          (bar_x + bar_len, y_pos + bar_h), bar_color, -1)
            cv2.putText(frame,
                        f"{name[:8]}: {prob:.1%}",
                        (bar_x + 4, y_pos + bar_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)
        return frame
