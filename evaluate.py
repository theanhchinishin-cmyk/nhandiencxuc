# -*- coding: utf-8 -*-
"""
evaluate.py - Chương trình đánh giá mô hình và vẽ ma trận nhầm lẫn (Confusion Matrix) trên tập kiểm thử Đông Nam Á (dataset/test_dung1landuynhat)
Sử dụng: python evaluate.py
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Nạp bộ máy cảm xúc EmotionEngine và danh sách 6 cảm xúc từ dự án chính (main.py)
from main import EmotionEngine, EMOTIONS

def run_evaluation():
    # 1. Khởi tạo bộ máy nhận diện cảm xúc
    # Bộ máy này tự động nạp mô hình HSEmotion và áp dụng bộ lọc CLAHE cũng như remapping
    eng = EmotionEngine()
    print("Khởi tạo bộ máy cảm xúc thành công!")
    print("Bộ ngưỡng thích nghi đang sử dụng:", eng.THRESH)
    
    # 2. Định nghĩa đường dẫn tập kiểm thử Đông Nam Á
    dataset_dir = os.path.join("dataset", "test_dung1landuynhat")
    
    if not os.path.exists(dataset_dir):
        print(f"Lỗi: Không tìm thấy thư mục tập kiểm thử '{dataset_dir}'!")
        return

    y_true = []  # Chứa nhãn cảm xúc THỰC TẾ của ảnh (nhãn đúng của thư mục)
    y_pred = []  # Chứa nhãn cảm xúc do AI DỰ ĐOÁN được

    print("\n--- Bắt đầu quét tập dữ liệu kiểm thử ---")
    
    # Duyệt qua từng thư mục cảm xúc trong 6 lớp cảm xúc
    for folder_name in EMOTIONS:
        folder_path = os.path.join(dataset_dir, folder_name)
        if not os.path.isdir(folder_path):
            print(f"Cảnh báo: Bỏ qua thư mục '{folder_name}' vì không tồn tại.")
            continue
            
        # Lấy danh sách tất cả các file ảnh trong thư mục đó
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Đang xử lý thư mục '{folder_name}' (có {len(files)} ảnh)...")
        
        for file in files:
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"  Lỗi đọc ảnh: {file}")
                continue
                
            # Đưa ảnh vào bộ xử lý (bao gồm tiền xử lý CLAHE và dự đoán của AI)
            scores = eng.predict(img)
            
            # Áp dụng logic quyết định cảm xúc (bao gồm bộ ngưỡng thích nghi)
            idx = np.argmax(scores)
            conf = float(scores[idx])
            pred_emo = EMOTIONS[idx]
            
            # Nếu độ tin cậy dưới ngưỡng nhạy của cảm xúc đó, đưa về nhãn mặc định Neutral (Bình thường)
            if conf < eng.THRESH.get(pred_emo, .4):
                pred_emo = 'Neutral'
                
            # Lưu lại nhãn thực tế và nhãn dự đoán dưới dạng số nguyên (chỉ số index)
            y_true.append(EMOTIONS.index(folder_name))
            y_pred.append(EMOTIONS.index(pred_emo))

    if not y_true:
        print("Lỗi: Không có ảnh nào được xử lý thành công!")
        return

    # 3. Tính toán báo cáo phân loại (Precision, Recall, F1-score)
    print("\n=== BÁO CÁO PHÂN LOẠI CHI TIẾT (CLASSIFICATION REPORT) ===")
    report = classification_report(y_true, y_pred, target_names=EMOTIONS, digits=4)
    print(report)

    # 4. Tính toán và vẽ Ma trận nhầm lẫn (Confusion Matrix)
    print("=== ĐANG VẼ MA TRẬN NHẦM LẪN (CONFUSION MATRIX) ===")
    cm = confusion_matrix(y_true, y_pred)
    
    # Vẽ biểu đồ nhiệt (Heatmap) của ma trận nhầm lẫn bằng thư viện Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    
    plt.title('Ma Trận Nhầm Lẫn (Confusion Matrix)')
    plt.ylabel('Cảm xúc thực tế (True Label)')
    plt.xlabel('Cảm xúc dự đoán (Predicted Label)')
    plt.tight_layout()
    
    # Tạo thư mục lưu báo cáo nếu chưa có
    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", "confusion_matrix.png")
    
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Lưu thành công biểu đồ ma trận nhầm lẫn vào: {out_path}")

if __name__ == '__main__':
    run_evaluation()
