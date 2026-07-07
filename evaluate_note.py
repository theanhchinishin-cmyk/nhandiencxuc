# -*- coding: utf-8 -*-
"""
evaluate.py - Chuong trinh danh gia mo hinh tren tap kiem thu Dong Nam A
Su dung: python evaluate.py
"""

import os  # [Dòng 7] Nạp thư viện os để quản lý đường dẫn thư mục ảnh kiểm thử trên ổ cứng máy tính.
import numpy as np  # [Dòng 8] Nạp numpy để thực hiện phép toán tìm vị trí cảm xúc lớn nhất (hàm argmax).
import cv2  # [Dòng 9] Nạp thư viện OpenCV (cv2) để đọc các file ảnh kiểm thử trắng đen/màu từ ổ cứng vào bộ nhớ.
import matplotlib.pyplot as plt  # [Dòng 10] Nạp thư viện vẽ biểu đồ matplotlib để thiết kế và lưu biểu đồ ma trận nhầm lẫn.
import seaborn as sns  # [Dòng 11] Nạp thư viện seaborn để vẽ biểu đồ nhiệt (Heatmap) của ma trận nhầm lẫn với màu sắc trực quan.
from sklearn.metrics import classification_report, confusion_matrix  # [Dòng 12] Nạp 2 hàm báo cáo đo lường (classification_report) và ma trận nhầm lẫn (confusion_matrix) của sklearn.

from main import EmotionEngine, EMOTIONS  # [Dòng 14] Nạp bộ máy EmotionEngine và danh sách 6 cảm xúc từ file chính main.py để đảm bảo đồng bộ xử lý.

def run_evaluation():  # [Dòng 16] Định nghĩa hàm run_evaluation thực hiện toàn bộ quy trình kiểm thử và vẽ báo cáo.
    eng = EmotionEngine()  # [Dòng 17] Khởi tạo bộ máy EmotionEngine (bộ máy này tự động nạp AI HSEmotion, bộ lọc CLAHE và các ngưỡng nhạy).
    dataset_dir = os.path.join("dataset", "test_dung1landuynhat")  # [Dòng 18] Thiết lập đường dẫn đến thư mục chứa 150 bức ảnh kiểm thử Đông Nam Á.
    
    if not os.path.exists(dataset_dir):  # [Dòng 20] Nếu đường dẫn không tồn tại (chưa có tập dữ liệu test), in thông báo lỗi và dừng chương trình.
        print(f"Error: Dataset directory '{dataset_dir}' not found!")  # [Dòng 21] In thông báo lỗi đường dẫn ra màn hình console.
        return  # [Dòng 22] Thoát hàm để bảo vệ chương trình không bị lỗi crash.

    y_true = []  # [Dòng 24] Khởi tạo mảng y_true trống để lưu lại danh sách nhãn cảm xúc đúng thực tế của ảnh.
    y_pred = []  # [Dòng 25] Khởi tạo mảng y_pred trống để lưu lại danh sách nhãn do AI dự đoán được.

    print("\n--- Scanning test dataset ---")  # [Dòng 27] In thông báo bắt đầu quét tập dữ liệu kiểm thử ra màn hình.
    for folder_name in EMOTIONS:  # [Dòng 28] Vòng lặp duyệt qua từng cảm xúc e trong 6 cảm xúc đích để quét các thư mục con tương ứng.
        folder_path = os.path.join(dataset_dir, folder_name)  # [Dòng 29] Tạo đường dẫn đầy đủ đến thư mục con của cảm xúc đó (ví dụ: dataset/test_dung1landuynhat/Happy).
        if not os.path.isdir(folder_path):  # [Dòng 30] Nếu thư mục con này không tồn tại, bỏ qua vòng lặp để tiếp tục quét thư mục khác.
            continue  # [Dòng 31] Bỏ qua vòng lặp hiện tại.
            
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]  # [Dòng 33] Lọc lấy tất cả các file có đuôi ảnh (.png, .jpg, .jpeg) trong thư mục con này.
        print(f"Processing '{folder_name}' ({len(files)} images)...")  # [Dòng 34] In tiến trình đang xử lý thư mục cảm xúc nào kèm số lượng ảnh ra màn hình console.
        
        for file in files:  # [Dòng 36] Vòng lặp duyệt qua từng tệp ảnh trong thư mục con đó.
            img_path = os.path.join(folder_path, file)  # [Dòng 37] Tạo đường dẫn đầy đủ đến tệp ảnh cụ thể đó.
            img = cv2.imread(img_path)  # [Dòng 38] Dùng OpenCV đọc bức ảnh từ ổ cứng đưa vào RAM dưới dạng ma trận điểm ảnh.
            if img is None:  # [Dòng 39] Nếu ảnh bị lỗi không đọc được (file hỏng), bỏ qua không xử lý ảnh này.
                continue  # [Dòng 40] Bỏ qua vòng lặp hiện tại để xét ảnh tiếp theo.
                
            scores = eng.predict(img)  # [Dòng 42] Đưa ảnh vào hàm predict của EmotionEngine để chạy cân bằng sáng CLAHE và được AI trả về 6 điểm số cảm xúc.
            idx = np.argmax(scores)  # [Dòng 43] Tìm vị trí (chỉ số index) của cảm xúc có điểm số lớn nhất.
            conf = float(scores[idx])  # [Dòng 44] Lấy điểm tin cậy (phần trăm) của cảm xúc lớn nhất đó.
            pred_emo = EMOTIONS[idx]  # [Dòng 45] Lấy tên chữ của cảm xúc dự đoán lớn nhất đó.
            
            if conf < eng.THRESH.get(pred_emo, .4):  # [Dòng 47] Kiểm tra với bộ ngưỡng nhạy thích nghi: nếu độ tin cậy thấp hơn ngưỡng quy định ở dòng 31 của main.py.
                pred_emo = 'Neutral'  # [Dòng 48] Ép cảm xúc dự đoán về nhãn mặc định Neutral (Bình thường) để tránh AI dự đoán bừa.
                
            y_true.append(EMOTIONS.index(folder_name))  # [Dòng 50] Lưu nhãn thực tế dạng số (chỉ số của thư mục con) vào mảng y_true.
            y_pred.append(EMOTIONS.index(pred_emo))  # [Dòng 51] Lưu nhãn dự đoán dạng số (chỉ số của cảm xúc AI đoán) vào mảng y_pred.

    print("\n=== CLASSIFICATION REPORT ===")  # [Dòng 53] In tiêu đề báo cáo phân loại ra màn hình console.
    report = classification_report(y_true, y_pred, target_names=EMOTIONS, digits=4)  # [Dòng 54] Gọi hàm classification_report để tự động tính toán Precision, Recall, F1-score của 6 lớp với độ chính xác 4 chữ số thập phân.
    print(report)  # [Dòng 55] In bảng báo cáo phân loại chi tiết ra màn hình đen console.

    print("=== PLOTTING CONFUSION MATRIX ===")  # [Dòng 57] In thông báo bắt đầu vẽ biểu đồ ma trận nhầm lẫn.
    cm = confusion_matrix(y_true, y_pred)  # [Dòng 58] Gọi hàm confusion_matrix của sklearn để tính toán ma trận nhầm lẫn giữa nhãn đúng và nhãn đoán.
    
    plt.figure(figsize=(8, 6))  # [Dòng 60] Tạo khung vẽ biểu đồ có kích thước rộng 8 inch, cao 6 inch.
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',   # [Dòng 61] Dùng Seaborn vẽ biểu đồ nhiệt (Heatmap) của ma trận: annot=True để in số lượng trực tiếp lên ô, cmap='Blues' tô màu xanh dương.
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)  # [Dòng 62] xticklabels và yticklabels giúp ghi tên các chữ cảm xúc lên hai trục ngang X và dọc Y của ma trận.
    
    plt.title('Ma Tran Nham Lan (Confusion Matrix)')  # [Dòng 64] Đặt tiêu đề cho biểu đồ ma trận nhầm lẫn là 'Ma Trận Nhầm Lẫn (Confusion Matrix)'.
    plt.ylabel('Cam xuc thuc te (True Label)')  # [Dòng 65] Đặt nhãn trục dọc Y biểu thị cảm xúc đúng thực tế (True Label).
    plt.xlabel('Cam xuc du doan (Predicted Label)')  # [Dòng 66] Đặt nhãn trục ngang X biểu thị cảm xúc do AI dự đoán (Predicted Label).
    plt.tight_layout()  # [Dòng 67] Tự động căn chỉnh lề biểu đồ để chữ không bị đè lên nhau.
    
    os.makedirs("reports", exist_ok=True)  # [Dòng 69] Tạo thư mục reports/ nếu thư mục này chưa có sẵn trên máy tính.
    out_path = os.path.join("reports", "confusion_matrix.png")  # [Dòng 70] Thiết lập đường dẫn lưu tệp ảnh ma trận nhầm lẫn là reports/confusion_matrix.png.
    plt.savefig(out_path, dpi=150)  # [Dòng 71] Lưu biểu đồ ma trận nhiệt thành file ảnh chất lượng cao 150 DPI.
    plt.close()  # [Dòng 72] Đóng hình vẽ matplotlib để giải phóng bộ nhớ RAM.
    print(f"Saved confusion matrix plot to: {out_path}")  # [Dòng 73] In thông báo đã lưu tệp ảnh ma trận nhầm lẫn thành công ra màn hình console.

if __name__ == '__main__':  # [Dòng 75] Điểm vào khởi chạy chính thức của chương trình Python khi gõ lệnh chạy từ terminal.
    run_evaluation()  # [Dòng 76] Gọi hàm run_evaluation để thực hiện quy trình đánh giá.
