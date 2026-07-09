# -*- coding: utf-8 -*-
"""
evaluate.py - Danh gia doi chung: Mo hinh thiet lap 1 vs Thiet lap 2
Su dung: python evaluate.py
"""

import os  # [Dòng 7] Nạp thư viện os để quản lý đường dẫn thư mục ảnh kiểm thử trên ổ cứng máy tính.
import numpy as np  # [Dòng 8] Nạp numpy để thực hiện phép toán tìm vị trí cảm xúc lớn nhất (hàm argmax).
import cv2  # [Dòng 9] Nạp thư viện OpenCV (cv2) để đọc các file ảnh kiểm thử trắng đen/màu từ ổ cứng vào bộ nhớ.
import matplotlib.pyplot as plt  # [Dòng 10] Nạp thư viện vẽ biểu đồ matplotlib để thiết kế và lưu biểu đồ ma trận nhầm lẫn.
import seaborn as sns  # [Dòng 11] Nạp thư viện seaborn để vẽ biểu đồ nhiệt (Heatmap) của ma trận nhầm lẫn với màu sắc trực quan.
from sklearn.metrics import classification_report, confusion_matrix  # [Dòng 12] Nạp 2 hàm báo cáo đo lường (classification_report) và ma trận nhầm lẫn (confusion_matrix) của sklearn.

from main import EmotionEngine, EMOTIONS  # [Dòng 14] Nạp bộ máy EmotionEngine và danh sách 6 cảm xúc từ file chính main.py để đảm bảo đồng bộ xử lý.

def run_evaluation():  # [Dòng 16] Định nghĩa hàm run_evaluation thực hiện toàn bộ quy trình kiểm thử đối chứng và vẽ báo cáo.
    eng = EmotionEngine()  # [Dòng 17] Khởi tạo bộ máy EmotionEngine (bộ máy này tự động nạp AI HSEmotion, bộ lọc CLAHE và các ngưỡng nhạy).
    dataset_dir = os.path.join("dataset", "test_dung1landuynhat")  # [Dòng 18] Thiết lập đường dẫn đến thư mục chứa 150 bức ảnh kiểm thử Đông Nam Á.
    
    if not os.path.exists(dataset_dir):  # [Dòng 20] Nếu đường dẫn không tồn tại (chưa có tập dữ liệu test), in thông báo lỗi và dừng chương trình.
        print(f"Error: Dataset directory '{dataset_dir}' not found!")  # [Dòng 21] In thông báo lỗi đường dẫn ra màn hình console.
        return  # [Dòng 22] Thoát hàm để bảo vệ chương trình không bị lỗi crash.

    y_true_raw = []  # [Dòng 24] Khởi tạo mảng y_true_raw trống để lưu nhãn thực tế cho Thiết lập 1 (Ảnh thô).
    y_pred_raw = []  # [Dòng 25] Khởi tạo mảng y_pred_raw trống để lưu nhãn AI dự đoán cho Thiết lập 1 (Ảnh thô).
    
    y_true_crop = []  # [Dòng 27] Khởi tạo mảng y_true_crop trống để lưu nhãn thực tế cho Thiết lập 2 (Có cắt mặt).
    y_pred_crop = []  # [Dòng 28] Khởi tạo mảng y_pred_crop trống để lưu nhãn AI dự đoán cho Thiết lập 2 (Có cắt mặt).

    print("\n--- Scanning test dataset ---")  # [Dòng 30] In thông báo bắt đầu quét tập dữ liệu kiểm thử ra màn hình.
    for folder_name in EMOTIONS:  # [Dòng 31] Vòng lặp duyệt qua từng cảm xúc e trong 6 cảm xúc đích để quét các thư mục con tương ứng.
        folder_path = os.path.join(dataset_dir, folder_name)  # [Dòng 32] Tạo đường dẫn đầy đủ đến thư mục con của cảm xúc đó (ví dụ: dataset/test_dung1landuynhat/Happy).
        if not os.path.isdir(folder_path):  # [Dòng 33] Nếu thư mục con này không tồn tại, bỏ qua vòng lặp để tiếp tục quét thư mục khác.
            continue  # [Dòng 34] Bỏ qua vòng lặp hiện tại.
            
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]  # [Dòng 36] Lọc lấy tất cả các file có đuôi ảnh (.png, .jpg, .jpeg) trong thư mục con này.
        print(f"Processing '{folder_name}' ({len(files)} images)...")  # [Dòng 37] In tiến trình đang xử lý thư mục cảm xúc nào kèm số lượng ảnh ra màn hình console.
        
        for file in files:  # [Dòng 39] Vòng lặp duyệt qua từng tệp ảnh trong thư mục con đó.
            img_path = os.path.join(folder_path, file)  # [Dòng 40] Tạo đường dẫn đầy đủ đến tệp ảnh cụ thể đó.
            img = cv2.imread(img_path)  # [Dòng 41] Dùng OpenCV đọc bức ảnh từ ổ cứng đưa vào RAM dưới dạng ma trận điểm ảnh.
            if img is None:  # [Dòng 42] Nếu ảnh bị lỗi không đọc được (file hỏng), bỏ qua không xử lý ảnh này.
                continue  # [Dòng 43] Bỏ qua vòng lặp hiện tại để xét ảnh tiếp theo.
                
            scores_raw = eng.predict(img)  # [Dòng 45] Đoạn 1: Đưa thẳng ảnh thô nguyên bản không qua cắt lọc vào hàm dự đoán predict.
            idx_raw = np.argmax(scores_raw)  # [Dòng 46] Tìm vị trí (chỉ số index) của cảm xúc có điểm số lớn nhất cho ảnh thô.
            conf_raw = float(scores_raw[idx_raw])  # [Dòng 47] Lấy điểm tin cậy (phần trăm) của cảm xúc lớn nhất cho ảnh thô.
            pred_raw = EMOTIONS[idx_raw]  # [Dòng 48] Lấy tên chữ của cảm xúc dự đoán lớn nhất cho ảnh thô.
            if conf_raw < eng.THRESH.get(pred_raw, .4):  # [Dòng 49] Kiểm tra với bộ ngưỡng nhạy thích nghi: nếu độ tin cậy thấp hơn ngưỡng, ép về Neutral.
                pred_raw = 'Neutral'  # [Dòng 50] Ép cảm xúc dự đoán về nhãn mặc định Neutral (Bình thường).
            y_true_raw.append(EMOTIONS.index(folder_name))  # [Dòng 51] Lưu nhãn thực tế dạng số cho Thiết lập 1 (Ảnh thô) vào mảng y_true_raw.
            y_pred_raw.append(EMOTIONS.index(pred_raw))  # [Dòng 52] Lưu nhãn dự đoán dạng số cho Thiết lập 1 (Ảnh thô) vào mảng y_pred_raw.
            
            faces = eng.detect(img)  # [Dòng 54] Đoạn 2: Dùng Haar Cascade dò tìm xem trong bức ảnh này có chứa khuôn mặt nào không.
            if len(faces) > 0:  # [Dòng 55] Nếu tìm thấy ít nhất một khuôn mặt trong ảnh.
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)  # [Dòng 56] Sắp xếp chọn ra khuôn mặt có diện tích (rộng x cao) lớn nhất để xử lý.
                x, y, w, h = faces[0]  # [Dòng 57] Rút ra tọa độ x, y và chiều rộng w, chiều cao h của khuôn mặt lớn nhất đó.
                face_img = img[y:y+h, x:x+w]  # [Dòng 58] Thực hiện cắt lát ma trận ảnh để lấy riêng vùng khuôn mặt lưu vào biến face_img.
            else:  # [Dòng 59] Trường hợp không dò tìm thấy khuôn mặt nào trong ảnh (ví dụ do ảnh mờ hoặc chụp nghiêng).
                face_img = img  # [Dòng 60] Sử dụng giải pháp dự phòng: coi toàn bộ bức ảnh gốc chính là vùng khuôn mặt.
                
            scores_crop = eng.predict(face_img)  # [Dòng 62] Đưa ảnh mặt đã cắt lọc vào hàm dự đoán predict của EmotionEngine.
            idx_crop = np.argmax(scores_crop)  # [Dòng 63] Tìm vị trí (chỉ số index) của cảm xúc có điểm số lớn nhất cho ảnh đã cắt mặt.
            conf_crop = float(scores_crop[idx_crop])  # [Dòng 64] Lấy điểm tin cậy (phần trăm) của cảm xúc lớn nhất cho ảnh đã cắt mặt.
            pred_crop = EMOTIONS[idx_crop]  # [Dòng 65] Lấy tên chữ của cảm xúc dự đoán lớn nhất cho ảnh đã cắt mặt.
            if conf_crop < eng.THRESH.get(pred_crop, .4):  # [Dòng 66] Kiểm tra với bộ ngưỡng nhạy thích nghi cho ảnh đã cắt mặt, nếu thấp hơn ngưỡng, ép về Neutral.
                pred_crop = 'Neutral'  # [Dòng 67] Ép cảm xúc dự đoán về nhãn mặc định Neutral (Bình thường) cho thiết lập 2.
            y_true_crop.append(EMOTIONS.index(folder_name))  # [Dòng 68] Lưu nhãn thực tế dạng số cho Thiết lập 2 (Cắt mặt) vào mảng y_true_crop.
            y_pred_crop.append(EMOTIONS.index(pred_crop))  # [Dòng 69] Lưu nhãn dự đoán dạng số cho Thiết lập 2 (Cắt mặt) vào mảng y_pred_crop.

    print("\n" + "="*50)  # [Dòng 71] In thanh phân cách tiêu đề báo cáo cho Thiết lập 1.
    print("=== THIET LAP 1: ANH GO THO (KHONG QUA CAT MAT) ===")  # [Dòng 72] In tiêu đề báo cáo phân loại Thiết lập 1: Ảnh gốc thô (không qua cắt mặt).
    print("="*50)  # [Dòng 73] In thanh phân cách tiêu đề báo cáo.
    report_raw = classification_report(y_true_raw, y_pred_raw, target_names=EMOTIONS, digits=4)  # [Dòng 74] Gọi hàm classification_report để tự động tính toán Precision, Recall, F1-score của Thiết lập 1.
    print(report_raw)  # [Dòng 75] In bảng báo cáo phân loại chi tiết của Thiết lập 1 ra màn hình console.

    print("\n" + "="*50)  # [Dòng 77] In thanh phân cách tiêu đề báo cáo cho Thiết lập 2.
    print("=== THIET LAP 2: KET HOP BEN CAT MAT (HAAR CASCADE) ===")  # [Dòng 78] In tiêu đề báo cáo phân loại Thiết lập 2: Kết hợp bộ bám cắt mặt Haar Cascade.
    print("="*50)  # [Dòng 79] In thanh phân cách tiêu đề báo cáo.
    report_crop = classification_report(y_true_crop, y_pred_crop, target_names=EMOTIONS, digits=4)  # [Dòng 80] Gọi hàm classification_report để tự động tính toán các chỉ số đo lường của Thiết lập 2.
    print(report_crop)  # [Dòng 81] In bảng báo cáo phân loại chi tiết của Thiết lập 2 ra màn hình console.

    os.makedirs("reports", exist_ok=True)  # [Dòng 83] Tạo thư mục reports/ nếu thư mục này chưa có sẵn trên máy tính.
    
    cm_raw = confusion_matrix(y_true_raw, y_pred_raw)  # [Dòng 85] Tính toán ma trận nhầm lẫn cho Thiết lập 1 (Ảnh thô).
    plt.figure(figsize=(8, 6))  # [Dòng 86] Tạo khung vẽ biểu đồ có kích thước rộng 8 inch, cao 6 inch cho ma trận Thiết lập 1.
    sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTIONS, yticklabels=EMOTIONS)  # [Dòng 87] Dùng Seaborn vẽ biểu đồ nhiệt (Heatmap) của ma trận Thiết lập 1: annot=True, màu Blues.
    plt.title('Ma Tran Nham Lan (Thiet lap 1: Anh tho)')  # [Dòng 88] Đặt tiêu đề cho biểu đồ ma trận nhầm lẫn Thiết lập 1 là 'Ma Trận Nhầm Lẫn (Thiet lap 1: Anh tho)'.
    plt.ylabel('Cam xuc thuc te (True Label)')  # [Dòng 89] Đặt nhãn trục dọc Y biểu thị cảm xúc thực tế (True Label) cho thiết lập 1.
    plt.xlabel('Cam xuc du doan (Predicted Label)')  # [Dòng 90] Đặt nhãn trục ngang X biểu thị cảm xúc dự đoán (Predicted Label) cho thiết lập 1.
    plt.tight_layout()  # [Dòng 91] Tự động căn chỉnh lề biểu đồ thiết lập 1.
    plt.savefig(os.path.join("reports", "confusion_matrix_raw.png"), dpi=150)  # [Dòng 92] Lưu biểu đồ ma trận thô thành file ảnh reports/confusion_matrix_raw.png.
    plt.close()  # [Dòng 93] Đóng hình vẽ matplotlib để giải phóng bộ nhớ RAM.
    
    cm_crop = confusion_matrix(y_true_crop, y_pred_crop)  # [Dòng 95] Tính toán ma trận nhầm lẫn cho Thiết lập 2 (Co cắt mặt).
    plt.figure(figsize=(8, 6))  # [Dòng 96] Tạo khung vẽ biểu đồ có kích thước rộng 8 inch, cao 6 inch cho ma trận Thiết lập 2.
    sns.heatmap(cm_crop, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTIONS, yticklabels=EMOTIONS)  # [Dòng 97] Dùng Seaborn vẽ biểu đồ nhiệt của ma trận Thiết lập 2: annot=True, màu Blues.
    plt.title('Ma Tran Nham Lan (Thiet lap 2: Co cat mat)')  # [Dòng 98] Đặt tiêu đề cho biểu đồ ma trận nhầm lẫn Thiết lập 2 là 'Ma Trận Nhầm Lẫn (Thiet lap 2: Co cat mat)'.
    plt.ylabel('Cam xuc thuc te (True Label)')  # [Dòng 99] Đặt nhãn trục dọc Y cho thiết lập 2.
    plt.xlabel('Cam xuc du doan (Predicted Label)')  # [Dòng 100] Đặt nhãn trục ngang X cho thiết lập 2.
    plt.tight_layout()  # [Dòng 101] Tự động căn chỉnh lề biểu đồ thiết lập 2.
    plt.savefig(os.path.join("reports", "confusion_matrix_cropped.png"), dpi=150)  # [Dòng 102] Lưu biểu đồ ma trận cắt mặt thành file ảnh reports/confusion_matrix_cropped.png.
    plt.close()  # [Dòng 103] Đóng hình vẽ matplotlib để giải phóng bộ nhớ RAM.
    
    open(os.path.join("reports", "confusion_matrix.png"), 'wb').write(open(os.path.join("reports", "confusion_matrix_cropped.png"), 'rb').read())
    print("\nSaved confusion matrix plots to: reports/")  # [Dòng 106] Ghi đè tệp reports/confusion_matrix_cropped.png thành reports/confusion_matrix.png để đồng bộ báo cáo cũ.
  # [Dòng 107] In thông báo đã lưu tất cả các tệp ma trận nhầm lẫn thành công ra màn hình console.
if __name__ == '__main__':
    run_evaluation()  # [Dòng 109] Điểm vào khởi chạy chính thức của chương trình Python khi gõ lệnh chạy từ terminal.
