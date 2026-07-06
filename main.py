"""
Nhan dang cam xuc nguoi hoc online
Dung: python main.py --mode webcam
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

# Fix Unicode cho terminal Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import argparse
import cv2
import time
from datetime import datetime
from src.constants import EMOTION_COLORS


# ──────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────

def print_header(mode: str):
    print("  HE THONG NHAN DANG CAM XUC NGUOI HOC ONLINE")
    print(f"  Che do : {mode.upper()}")
    print(f"  Thoi diem: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n")


# ──────────────────────────────────────────────
#  MODE 1: WEBCAM
# ──────────────────────────────────────────────

def run_webcam(args):
    """nhan dien emotion truc tiep tu webcam"""
    from src.capture import CameraCapture
    from src.emotion_engine import EmotionEngine
    from src.emotion_stats import SessionStats

    print_header("WEBCAM")

    camera = CameraCapture(camera_id=args.camera)
    engine = EmotionEngine(model_name=args.model, weights_path=args.weights)

    if not camera.open_camera():
        print(" Không thể mở camera.")
        return

    print("\n  Phím điều khiển:")
    print("  • q / ESC  : Thoát")
    print("  • s        : Chụp ảnh màn hình")
    print("  • r        : Reset temporal smoothing buffer")
    print("\n Đang xử lý...\n")

    frame_count = 0
    skip_frames = 2  # bo qua 1 frame de tang FPS
    last_results = []
    start_time = time.time()

    # Thong ke cam xuc trong phien
    stats = SessionStats()

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret:
                break

            frame_count += 1

            # Frame bi skip: ve lai ket qua cu
            if frame_count % skip_frames != 0:
                display = _draw_webcam_overlay(frame, last_results, frame_count, start_time)
                camera.show_frame(display, "Emotion Detection")
                if _handle_key(camera, display, engine) == 'quit':
                    break
                continue

            # Frame xu ly AI
            results = engine.process_frame(frame)
            last_results = results

            # Ghi nhan thong ke cam xuc (append-only, O(1) -> ko anh huong FPS)
            elapsed = time.time() - start_time
            stats.record(elapsed, results)

            display = _draw_webcam_overlay(frame, results, frame_count, start_time)
            camera.show_frame(display, "Emotion Detection")

            if _handle_key(camera, display, engine) == 'quit':
                break

    except KeyboardInterrupt:
        print("\n Bi ngat boi nguoi dung (Ctrl+C)")
    finally:
        camera.release_camera()
        camera.destroy_all_windows()
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\n Ket thuc | {frame_count} frames | {elapsed:.1f}s | {fps:.1f} FPS")

        # Tao bao cao thong ke cam xuc
        stats.generate_report()


def _draw_webcam_overlay(frame, results, frame_count, start_time):
    """ve emotion box + info text len frame"""
    display = frame.copy()

    for result in results:
        x, y, w, h = result['bbox']
        emotion = result['emotion']
        confidence = result['confidence']
        color = EMOTION_COLORS.get(emotion, (255,255,255))

        cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
        label = f"{emotion}: {confidence:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(display, (x, y-lh-10), (x+lw+6, y), color, -1)
        cv2.putText(display, label, (x+3, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 2)

    # Info overlay goc tren trai
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    info = [
        f"Frame: {frame_count}",
        f"FPS:   {fps:.1f}",
        f"Faces: {len(results)}",
    ]
    for i, txt in enumerate(info):
        cv2.putText(display, txt, (10, 30 + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # Huong dan goc duoi
    cv2.putText(display, "q: thoat | s: chup anh | r: reset",
                (10, display.shape[0]-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

    return display


def _handle_key(camera, current_frame, engine):
    """xu ly phim bam. Tra ve 'quit' neu can thoat"""
    key = camera.wait_key(1) & 0xFF
    if key in (ord('q'), 27):
        print("\n Nguoi dung yeu cau thoat.")
        return 'quit'
    elif key == ord('s'):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"data/screenshot_{ts}.jpg"
        os.makedirs('data', exist_ok=True)
        cv2.imwrite(path, current_frame)
        print(f" Da luu anh: {path}")
    elif key == ord('r'):
        engine.face_buffers.clear()
        print(" Da reset temporal smoothing buffer.")
    return None


# ──────────────────────────────────────────────
#  MODE 2: IMAGE
# ──────────────────────────────────────────────

def run_image(args):
    """nhan dien emotion tu file anh"""
    from src.image_predictor import ImagePredictor

    print_header("IMAGE")

    if not args.input:
        print(" Can cung cap duong dan anh: --input <path>")
        return

    predictor = ImagePredictor(
        model_name=args.model,
        weights_path=args.weights
    )
    predictor.predict(
        image_path=args.input,
        output_path=args.output,
        show=True
    )


# ──────────────────────────────────────────────
#  MODE 3: VIDEO
# ──────────────────────────────────────────────

def run_video(args):
    """nhan dien emotion tu file video (.mp4, .avi, .mov, ...)"""
    import cv2
    from src.emotion_engine import EmotionEngine
    from src.emotion_stats import SessionStats

    print_header("VIDEO")

    if not args.input:
        print(" Can cung cap duong dan video: --input <path>")
        return

    if not os.path.isfile(args.input):
        print(f" Khong tim thay file: {args.input}")
        return

    engine = EmotionEngine(model_name=args.model, weights_path=args.weights)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f" Không thể mở video: {args.input}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_src      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration     = total_frames / fps_src if fps_src > 0 else 0

    print(f"\n Video: {args.input}")
    print(f"   Do phan giai : {width}x{height} | FPS goc: {fps_src:.1f}")
    print(f"   Tong so frame: {total_frames} (~{duration:.1f} giay)")

    # Khoi tao VideoWriter neu can luu video
    writer = None
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps_src, (width, height))
        print(f"   Luu ket qua : {args.output}")

    print(f"\n Dang xu ly...")
    print(f"  q / ESC : Dung  |  SPACE : Tam dung/Tiep tuc\n")

    frame_count = 0
    skip_frames = 2          # xu ly AI moi 2 frame
    last_results = []
    paused = False
    start_time = time.time()

    # Thong ke cam xuc trong video
    stats = SessionStats()

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n Da xu ly het video.")
                    break

                frame_count += 1

                # Chi chay AI moi skip_frames frame
                if frame_count % skip_frames == 0:
                    last_results = engine.process_frame(frame)
                    # Ghi nhan thong ke cam xuc (append-only)
                    elapsed = time.time() - start_time
                    stats.record(elapsed, last_results)

                display = _draw_video_overlay(
                    frame, last_results, frame_count, total_frames, fps_src
                )

                if writer:
                    writer.write(display)

                cv2.imshow("Emotion Detection — Video", display)

            # Xu ly phim bam
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                print("\n Dung theo yeu cau.")
                break
            elif key == ord(' '):
                paused = not paused
                print(f"  {'[Tam dung]' if paused else '[Tiep tuc]'}")
            elif key == ord('s'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"data/screenshot_{ts}.jpg"
                os.makedirs('data', exist_ok=True)
                cv2.imwrite(path, display)
                print(f"  Da luu anh: {path}")

    except KeyboardInterrupt:
        print("\n Bi ngat boi nguoi dung (Ctrl+C)")
    finally:
        cap.release()
        if writer:
            writer.release()
            print(f" Da luu video ket qua: {args.output}")
        cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        proc_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f" Ket thuc | {frame_count}/{total_frames} frames | {elapsed:.1f}s | {proc_fps:.1f} FPS")

        # Tao bao cao thong ke cam xuc cho video
        stats.generate_report()


def _draw_video_overlay(frame, results, frame_count, total_frames, src_fps):
    """ve emotion box + progress bar len frame video"""
    display = frame.copy()
    h, w = display.shape[:2]

    for result in results:
        x, y, bw, bh = result['bbox']
        emotion   = result['emotion']
        confidence = result['confidence']
        color = EMOTION_COLORS.get(emotion, (255,255,255))

        cv2.rectangle(display, (x, y), (x+bw, y+bh), color, 2)
        label = f"{emotion}: {confidence:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(display, (x, y-lh-10), (x+lw+6, y), color, -1)
        cv2.putText(display, label, (x+3, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 2)

    # Thanh tien trinh o duoi
    bar_h = 6
    progress = frame_count / total_frames if total_frames > 0 else 0
    cv2.rectangle(display, (0, h-bar_h), (w, h), (50,50,50), -1)
    cv2.rectangle(display, (0, h-bar_h), (int(w*progress), h), (0,200,255), -1)

    # Thong tin frame / thoi gian
    cur_sec  = frame_count / src_fps if src_fps > 0 else 0
    tot_sec  = total_frames / src_fps if src_fps > 0 else 0
    info_txt = (f"Frame {frame_count}/{total_frames}  "
                f"{cur_sec:.1f}s/{tot_sec:.1f}s  "
                f"Faces: {len(results)}")
    cv2.putText(display, info_txt, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # Huong dan nho
    guide = "q: thoat | SPACE: tam dung | s: chup anh"
    cv2.putText(display, guide, (10, h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)

    return display


# ──────────────────────────────────────────────
#  MODE 4: FINETUNE
# ──────────────────────────────────────────────

def run_finetune(args):
    """fine-tune model tren dataset tu thu thap"""
    from src.file_tuner import EmotionFineTuner

    print_header("FINE-TUNE")

    if not args.dataset:
        print(" Can cung cap duong dan dataset: --dataset <path>")
        return

    tuner = EmotionFineTuner(model_name=args.model)
    tuner.load_dataset(
        dataset_dir=args.dataset,
        batch_size=args.batch_size,
        test_split=0.0
    )
    history = tuner.train(
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.save_weights
    )

    # Luu training history de run_evaluate doc
    if history:
        import numpy as np
        history_path = args.save_weights.replace('.pt', '_history.npy')
        np.save(history_path, history, allow_pickle=True)
        print(f" Da luu lich su training: {history_path}")

    print(f"\n Buoc tiep theo:")
    print(f"   Danh gia: python main.py --mode evaluate --dataset {args.dataset}")
    print(f"   Demo:     python main.py --mode webcam --weights {args.save_weights}")


# ──────────────────────────────────────────────
#  MODE 5: EVALUATE
# ──────────────────────────────────────────────

def run_evaluate(args):
    """danh gia va so sanh model goc vs fine-tuned tren test set"""
    from src.file_tuner import EmotionFineTuner
    from src.evaluator import ModelEvaluator
    import torch

    print_header("EVALUATE")

    if not args.dataset:
        print(" Can cung cap duong dan dataset: --dataset <path>")
        return

    weights_path = args.weights or args.save_weights
    if not os.path.isfile(weights_path):
        print(f" Khong tim thay file fine-tuned weights: {weights_path}")
        print(f"   Chay fine-tune truoc: python main.py --mode finetune --dataset {args.dataset}")
        return

    # Load dataset (chi dung test set)
    print(" Dang load dataset...")
    tuner = EmotionFineTuner(model_name=args.model)
    tuner.load_dataset(dataset_dir=args.dataset, batch_size=args.batch_size, is_test_only=True)
    test_loader, class_names = tuner.test_data

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model goc
    print("\n Dang load model goc...")
    original_model = EmotionFineTuner._load_model(args.model).to(device)

    # Model fine-tuned
    print(" Dang load model fine-tuned...")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    finetuned_model = EmotionFineTuner._load_model(args.model).to(device)
    finetuned_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Danh gia
    evaluator = ModelEvaluator(class_names=class_names, output_dir='reports')

    # Doc history neu co
    history_path = weights_path.replace('.pt', '_history.npy')
    history = None
    if os.path.isfile(history_path):
        import numpy as np
        history = np.load(history_path, allow_pickle=True).item()

    evaluator.evaluate_and_compare(
        original_model=original_model,
        finetuned_model=finetuned_model,
        test_loader=test_loader,
        history=history
    )

    print(f"\n Ket qua da luu vao thu muc: reports/")


# ──────────────────────────────────────────────
#  ARGUMENT PARSER
# ──────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='He thong nhan dang cam xuc nguoi hoc online',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Vi du:
  python main.py --mode webcam
  python main.py --mode image   --input face.jpg
  python main.py --mode video   --input clip.mp4
  python main.py --mode video   --input clip.mp4 --output result.mp4
  python main.py --mode finetune --dataset dataset/ --epochs 15
  python main.py --mode evaluate --dataset dataset/
  python main.py --mode webcam  --weights models/finetuned.pt
        """
    )

    parser.add_argument('--mode', type=str, default='webcam',
                        choices=['webcam', 'image', 'video', 'finetune', 'evaluate'],
                        help='Che do chay (mac dinh: webcam)')

    # Model
    parser.add_argument('--model', type=str, default='enet_b0_8_best_afew',
                        choices=['enet_b0_8_best_afew', 'enet_b0_8_best_vgaf', 'enet_b2_8'],
                        help='Ten model HSEmotion base')
    parser.add_argument('--weights', type=str, default=None,
                        help='Duong dan fine-tuned weights .pt (webcam & image)')

    # Webcam
    parser.add_argument('--camera', type=int, default=0,
                        help='ID camera (mac dinh: 0)')

    # Image / Video mode
    parser.add_argument('--input', type=str, default=None,
                        help='Duong dan anh hoac video dau vao')
    parser.add_argument('--output', type=str, default=None,
                        help='Luu ket qua ra file (image: .jpg | video: .mp4)')

    # Fine-tune / Evaluate
    parser.add_argument('--dataset', type=str, default=None,
                        help='Thu muc dataset (mode finetune & evaluate)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='So epoch fine-tune (mac dinh: 15)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (mac dinh: 1e-4)')
    parser.add_argument('--batch-size', type=int, default=16, dest='batch_size',
                        help='Batch size (mac dinh: 16)')
    parser.add_argument('--save-weights', type=str, default='models/finetuned.pt',
                        dest='save_weights',
                        help='Duong dan luu model fine-tuned (mac dinh: models/finetuned.pt)')

    return parser


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        'webcam':   run_webcam,
        'image':    run_image,
        'video':    run_video,
        'finetune': run_finetune,
        'evaluate': run_evaluate,
    }

    try:
        dispatch[args.mode](args)
    except KeyboardInterrupt:
        print("\n\n Chuong trinh bi ngat boi nguoi dung.")
    except Exception as e:
        print(f"\n\n Loi: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
