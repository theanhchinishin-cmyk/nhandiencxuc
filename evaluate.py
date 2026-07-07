# -*- coding: utf-8 -*-
"""
evaluate.py - Chuong trinh danh gia mo hinh tren tap kiem thu Dong Nam A
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

    y_true = []
    y_pred = []

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
                
            scores = eng.predict(img)
            idx = np.argmax(scores)
            conf = float(scores[idx])
            pred_emo = EMOTIONS[idx]
            
            if conf < eng.THRESH.get(pred_emo, .4):
                pred_emo = 'Neutral'
                
            y_true.append(EMOTIONS.index(folder_name))
            y_pred.append(EMOTIONS.index(pred_emo))

    if not y_true:
        print("Error: No images were successfully processed!")
        return

    print("\n=== CLASSIFICATION REPORT ===")
    report = classification_report(y_true, y_pred, target_names=EMOTIONS, digits=4)
    print(report)

    print("=== PLOTTING CONFUSION MATRIX ===")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    
    plt.title('Ma Tran Nham Lan (Confusion Matrix)')
    plt.ylabel('Cam xuc thuc te (True Label)')
    plt.xlabel('Cam xuc du doan (Predicted Label)')
    plt.tight_layout()
    
    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", "confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to: {out_path}")

if __name__ == '__main__':
    run_evaluation()
