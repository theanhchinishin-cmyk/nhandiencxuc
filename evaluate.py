# -*- coding: utf-8 -*-
"""
evaluate.py - Danh gia doi chung: Mo hinh thiet lap 1 vs Thiet lap 2
Su dung: python evaluate.py
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from main import EmotionEngine, EMOTIONS

def run_evaluation():
    eng = EmotionEngine()
    dataset_dir = os.path.join("dataset", "test_dung1landuynhat")
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found!")
        return

    y_true_raw = []
    y_pred_raw = []
    
    y_true_crop = []
    y_pred_crop = []

    print("\n--- Scanning test dataset ---")
    for folder_name in EMOTIONS:
        folder_path = os.path.join(dataset_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Processing '{folder_name}' ({len(files)} images)...")
        
        for file in files:
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            scores_raw = eng.predict(img)
            idx_raw = np.argmax(scores_raw)
            conf_raw = float(scores_raw[idx_raw])
            pred_raw = EMOTIONS[idx_raw]
            if conf_raw < eng.THRESH.get(pred_raw, .4):
                pred_raw = 'Neutral'
            y_true_raw.append(EMOTIONS.index(folder_name))
            y_pred_raw.append(EMOTIONS.index(pred_raw))
            
            faces = eng.detect(img)
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                x, y, w, h = faces[0]
                face_img = img[y:y+h, x:x+w]
            else:
                face_img = img
                
            scores_crop = eng.predict(face_img)
            idx_crop = np.argmax(scores_crop)
            conf_crop = float(scores_crop[idx_crop])
            pred_crop = EMOTIONS[idx_crop]
            if conf_crop < eng.THRESH.get(pred_crop, .4):
                pred_crop = 'Neutral'
            y_true_crop.append(EMOTIONS.index(folder_name))
            y_pred_crop.append(EMOTIONS.index(pred_crop))

    print("\n" + "="*50)
    print("=== THIET LAP 1: ANH GO THO (KHONG QUA CAT MAT) ===")
    print("="*50)
    report_raw = classification_report(y_true_raw, y_pred_raw, target_names=EMOTIONS, digits=4)
    print(report_raw)

    print("\n" + "="*50)
    print("=== THIET LAP 2: KET HOP BEN CAT MAT (HAAR CASCADE) ===")
    print("="*50)
    report_crop = classification_report(y_true_crop, y_pred_crop, target_names=EMOTIONS, digits=4)
    print(report_crop)

    os.makedirs("reports", exist_ok=True)
    
    cm_raw = confusion_matrix(y_true_raw, y_pred_raw)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title('Ma Tran Nham Lan (Thiet lap 1: Anh tho)')
    plt.ylabel('Cam xuc thuc te (True Label)')
    plt.xlabel('Cam xuc du doan (Predicted Label)')
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "confusion_matrix_raw.png"), dpi=150)
    plt.close()
    
    cm_crop = confusion_matrix(y_true_crop, y_pred_crop)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_crop, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title('Ma Tran Nham Lan (Thiet lap 2: Co cat mat)')
    plt.ylabel('Cam xuc thuc te (True Label)')
    plt.xlabel('Cam xuc du doan (Predicted Label)')
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "confusion_matrix_cropped.png"), dpi=150)
    plt.close()
    
    open(os.path.join("reports", "confusion_matrix.png"), 'wb').write(open(os.path.join("reports", "confusion_matrix_cropped.png"), 'rb').read())
    print("\nSaved confusion matrix plots to: reports/")

if __name__ == '__main__':
    run_evaluation()
