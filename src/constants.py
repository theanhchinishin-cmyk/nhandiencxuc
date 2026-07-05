"""Mau sac va hang so dung chung"""

# Mau BGR cho tung emotion (OpenCV format)
EMOTION_COLORS = {
    'Happy':    (0, 255, 0),      # Xanh lá
    'Sad':      (255, 80, 80),    # Đỏ nhạt
    'Angry':    (0, 0, 255),      # Đỏ
    'Surprise': (0, 165, 255),    # Cam
    'Neutral':  (180, 180, 180),  # Xám
    'Fear':     (180, 0, 180),    # Tím
    'Disgust':  (0, 160, 160),    # Xanh lục
}

# Map ten emotion tu model (8 classes) sang ten hien thi
EMOTION_DISPLAY_MAP = {
    'Anger': 'Angry',
    'Happiness': 'Happy',
    'Sadness': 'Sad',
    'Contempt': 'Disgust',
}

# 8 emotion labels (theo thu tu output cua model)
EMOTION_LABELS_8 = [
    'Anger', 'Contempt', 'Disgust', 'Fear',
    'Happiness', 'Neutral', 'Sadness', 'Surprise'
]

# 7 labels (bo Contempt) cho thu thap data
EMOTION_LABELS_7 = [
    'Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise'
]
