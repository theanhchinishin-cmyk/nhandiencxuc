"""Thong ke va truc quan hoa cam xuc trong phien lam viec

Tinh nang:
  - Ghi lai lich su cam xuc append-only (ko anh huong FPS)
  - Tao bao cao gom Pie Chart + Timeline Chart
  - Tu dong luu file + show man hinh khi ket thuc phien
"""

import os
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict, Optional
from src.constants import EMOTION_COLORS


class SessionStats:
    """Ghi lai lich su cam xuc nhan dang duoc trong 1 phien

    Co che hoat dong:
      - buffer list append-only -> O(1) record, ko block AI loop
      - generate_report() duoc goi 1 lan khi session ket thuc
      - Pie chart: phan bo % tung emotion
      - Timeline: scatter + step-line, X=thoi gian, Y=cam xuc

    Example:
        stats = SessionStats()
        while running:
            results = engine.process_frame(frame)
            stats.record(elapsed_time, results)
        stats.generate_report()
    """

    def __init__(self):
        # buffer append-only: list cac dict {time, emotion, confidence}
        self.history: List[Dict] = []

        # Thu tu emotion co dinh de ve bieu do nhat quan
        # Dua tren key insertion order cua EMOTION_COLORS (Python 3.7+)
        self._emotion_order = list(EMOTION_COLORS.keys())

    # ──────────────────────────────────────────────
    #  RECORD — ghi nhan 1 mau du lieu
    # ──────────────────────────────────────────────

    def record(self, elapsed_sec: float, results: List[Dict]) -> None:
        """Ghi trang thai cam xuc tai thoi diem hien tai

        Args:
            elapsed_sec: so giay da troi qua tu khi bat dau session
            results:     ket qua tu EmotionEngine.process_frame()
                         List[{'bbox','emotion','confidence','probabilities'}]

        Ghi chu:
            - Chi lay face dau tien (primary face) lam dai dien
            - Neu frame khong co face (results = []) -> bo qua
            - Append vao list -> O(1), ko anh huong FPS
        """
        if not results:
            return  # khong co face -> khong ghi nhan

        # Lay face dau tien lam primary
        primary = results[0]
        self.history.append({
            'time':       elapsed_sec,
            'emotion':    primary['emotion'],
            'confidence': primary['confidence']
        })

    # ──────────────────────────────────────────────
    #  REPORT — sinh bao cao truc quan
    # ──────────────────────────────────────────────

    def generate_report(self, output_dir: str = 'reports') -> None:
        """Tao va luu bao cao thong ke cam xuc (Pie + Timeline)

        Xu ly:
            - BGR -> RGB chuan mau cho matplotlib
            - Pie chart: % va so luong tung emotion
            - Timeline: diem scatter (size=confidence) + step line
            - Luu file PNG + show man hinh (block cho user xem)
            - History rong -> in warning, khong tao chart
            - Tat ca emotion = 1 -> pie chart 100%, timeline flat line
        """
        if not self.history:
            print("  [Stats] Khong co du lieu thong ke (session rong).")
            return

        # Dam bao thu muc output ton tai
        os.makedirs(output_dir, exist_ok=True)

        # ── Giai nen du lieu ─────────────────────
        timestamps  = [h['time']       for h in self.history]
        emotions    = [h['emotion']    for h in self.history]
        confidences = [h['confidence'] for h in self.history]

        # Dem tan suat tung emotion
        emotion_counts = Counter(emotions)

        # Chi lay emotion co xuat hien (giu nguyen thu tu _emotion_order)
        present_emotions = [e for e in self._emotion_order if e in emotion_counts]
        counts = [emotion_counts[e] for e in present_emotions]

        # Chuyen mau BGR (OpenCV) -> RGB (matplotlib) -> normalized [0,1]
        colors_rgb = []
        for e in present_emotions:
            bgr = EMOTION_COLORS[e]          # (B, G, R)
            rgb = tuple(c / 255.0 for c in reversed(bgr))  # (R/255, G/255, B/255)
            colors_rgb.append(rgb)

        # Map emotion name -> int cho truc Y cua timeline
        emotion_to_idx = {e: i for i, e in enumerate(self._emotion_order)}
        y_vals = [emotion_to_idx[e] for e in emotions]

        # ── Tao figure layout ────────────────────
        # 2 subplots doc: pie (tren) + timeline (duoi)
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(12, 10),
            gridspec_kw={'height_ratios': [1, 1.5]}
        )
        fig.suptitle(
            'Bao Cao Thong Ke Cam Xuc — Online Learner Emotion Recognition',
            fontsize=14, fontweight='bold', y=0.98
        )

        # ── 1. PIE CHART ─────────────────────────
        total_records = len(emotions)
        wedges, texts, autotexts = ax1.pie(
            counts,
            labels=present_emotions,
            colors=colors_rgb,
            autopct=lambda p: f'{p:.1f}%\n({int(round(p * total_records / 100))}/{total_records})',
            startangle=90,
            textprops={'fontsize': 11},
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
        )
        # Tang kich co chu % cho de doc
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight('bold')
        ax1.set_title('Phan Bo Cam Xuc (Pie Chart)', fontsize=12, pad=15)

        # ── 2. TIMELINE LINE CHART ───────────────
        # Step line: the hien su chuyen doi cam xuc
        ax2.step(
            timestamps, y_vals,
            where='post', color='gray', alpha=0.5,
            linewidth=1.5, linestyle='--'
        )

        # Scatter: moi cham la 1 frame, size = confidence
        for t, e, c in zip(timestamps, emotions, confidences):
            idx = emotion_to_idx[e]
            bgr = EMOTION_COLORS[e]
            rgb = tuple(v / 255.0 for v in reversed(bgr))
            ax2.scatter(
                t, idx,
                s=max(40, c * 250),     # size ty le voi confidence (min 40)
                color=[rgb],
                edgecolors='black',
                linewidth=0.5,
                zorder=5                # ve len tren step line
            )

        # Format truc Y: emotion names
        ax2.set_yticks(range(len(self._emotion_order)))
        ax2.set_yticklabels(self._emotion_order, fontsize=10)
        ax2.set_ylim(-0.5, len(self._emotion_order) - 0.5)

        # Format truc X
        ax2.set_xlabel('Thoi gian (giay)', fontsize=11)
        ax2.set_ylabel('Cam xuc', fontsize=11)
        ax2.set_title('Dien Bieu Cam Xuc Theo Thoi Gian (Timeline)', fontsize=12, pad=10)
        ax2.grid(True, alpha=0.3, linestyle=':')

        # ── Session summary ──────────────────────
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        avg_rate = len(timestamps) / duration if duration > 0 else 0
        summary_txt = (
            f"Session: {duration:.0f}s | {total_records} records "
            f"| {avg_rate:.1f} record/s\n"
            f"Emotions xuat hien: {len(present_emotions)}/6 loai"
        )
        fig.text(
            0.5, 0.01, summary_txt,
            ha='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8)
        )

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        # ── Luu file ─────────────────────────────
        output_path = os.path.join(output_dir, 'emotion_report.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  [Stats] Da luu bao ca: {output_path}")

        # ── Hien thi len man hinh ────────────────
        # block=True: user xem xong dong cua so -> chuong trinh thoat
        plt.show(block=True)
        plt.close()
