"""Camera + capture video"""

import cv2
import numpy as np
from typing import Tuple, Optional
from src.constants import EMOTION_COLORS

class CameraCapture:
    """class quan ly camera"""

    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.is_opened = False

    def open_camera(self) -> bool:
        """mo camera, tra ve True neu ok"""
        # uu tien DirectShow (on dinh hon MSMF tren Windows)
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            # fallback: thu backend mac dinh
            self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            print(f" KHONG THE MO CAMERA ID: {self.camera_id}")
            return False

        # set resolution 640x480
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.is_opened = True
        print(f" Da mo camera ID: {self.camera_id}")
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """doc 1 frame tu camera, tra ve (ok, frame)"""
        if not self.is_opened or self.cap is None:
            return False, None

        ret, frame = self.cap.read()

        if not ret:
            print("Khong doc duoc frame tu camera")
            return False, None

        return True, frame

    def release_camera(self):
        """tat camera, giai phong tai nguyen"""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            print(" Da dong camera")

    def draw_text(self, frame: np.ndarray, text: str,
                  position: Tuple[int, int] = (10, 30),
                  font_scale: float = 0.7,
                  color: Tuple[int, int, int] = (0, 255, 0),
                  thickness: int = 2) -> np.ndarray:
        """ve text len frame"""
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # ve background cho de doc
        cv2.rectangle(
            frame,
            (position[0] - 5, position[1] - text_height - 5),
            (position[0] + text_width + 5, position[1] + 5),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame, text, position,
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness
        )
        return frame

    def draw_emotion_box(self, frame: np.ndarray,
                         emotion: str, confidence: float,
                         bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """ve khung quanh mat + emotion"""
        x, y, w, h = bbox
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        label = f"{emotion}: {confidence:.2f}"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def show_frame(self, frame: np.ndarray, window_name: str = "Emotion Detection"):
        """show frame len man hinh"""
        cv2.imshow(window_name, frame)

    def wait_key(self, delay: int = 1) -> int:
        """cho nhan phim, delay=1ms"""
        return cv2.waitKey(delay)

    @staticmethod
    def destroy_all_windows():
        """dong het cua so OpenCV"""
        cv2.destroyAllWindows()

    def get_frame_size(self) -> Tuple[int, int]:
        """lay width, height hien tai"""
        if not self.is_opened or self.cap is None:
            return (0, 0)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def get_fps(self) -> float:
        """lay FPS cua camera"""
        if not self.is_opened or self.cap is None:
            return 0.0
        return self.cap.get(cv2.CAP_PROP_FPS)