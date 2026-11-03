r"""
Script kiểm thử Thuật toán Hình học Độ sâu (Hướng 2):
Phát hiện BỨC TƯỜNG (Wall Plane) & BẬC TAM CẤP (Stairs/Steps).

Chạy (từ thư mục gốc):
    .\.venv\Scripts\python.exe backend\scripts\test_geometric_hazard.py
"""

import io
import os
import sys
import time
import cv2
import numpy as np

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from app.vision.geometric_hazard_detector import geometric_hazard_detector


def create_synthetic_wall_and_floor(w=640, h=480, wall_dist=1.4):
    """Tạo bản đồ độ sâu giả lập: nửa dưới là sàn nhà, nửa trên gặp bức tường chắn ngang cách 1.4m."""
    dmap = np.zeros((h, w), dtype=np.float32)
    # Nửa dưới: sàn nhà, khoảng cách tăng dần từ 0.5m (dưới chân) lên 1.4m (chân tường)
    for y in range(int(h * 0.65), h):
        ratio = (h - y) / (h * 0.35)
        dmap[y, :] = 0.5 + (1.0 - ratio) * 0.9  # 0.5m -> 1.4m

    # Nửa giữa và trên: bức tường thẳng đứng cách 1.4m (thêm nhiễu nhẹ)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.02, size=(int(h * 0.65), w)).astype(np.float32)
    dmap[:int(h * 0.65), :] = wall_dist + noise
    return dmap


def create_synthetic_stairs(w=640, h=480):
    """Tạo ảnh và bản đồ độ sâu giả lập 3 bậc tam cấp ở nửa dưới."""
    frame = np.full((h, w, 3), 180, dtype=np.uint8)
    dmap = np.zeros((h, w), dtype=np.float32)

    # 3 bậc tam cấp tại các vị trí Y: 300, 360, 420
    step_ys = [int(h * 0.58), int(h * 0.70), int(h * 0.82)]
    for y in step_ys:
        cv2.line(frame, (int(w * 0.1), y), (int(w * 0.9), y), (40, 40, 40), 3)

    # Bản đồ độ sâu nhảy bậc
    cur_z = 2.2
    prev_y = 0
    for y in step_ys:
        dmap[prev_y:y, :] = cur_z
        cur_z -= 0.35  # Mỗi bậc cao lên, gần lại 0.35m
        prev_y = y
    dmap[prev_y:, :] = 0.8
    return frame, dmap


def test_geometric_detector():
    print("==================================================")
    print("🚀 BẮT ĐẦU KIỂM THỬ THUẬT TOÁN HÌNH HỌC (HƯỚNG 2)")
    print("==================================================")

    # 1. TEST PHÁT HIỆN BỨC TƯỜNG (WALL)
    wall_dmap = create_synthetic_wall_and_floor(wall_dist=1.3)
    t0 = time.time()
    wall_res = geometric_hazard_detector.detect_wall(wall_dmap)
    t_wall_ms = (time.time() - t0) * 1000

    print(f"\n[1] TEST BỨC TƯỜNG:")
    print(f"    - Thời gian tính toán: {t_wall_ms:.2f} ms")
    if wall_res:
        print(f"    - Kết quả: ✅ PHÁT HIỆN {wall_res['class_name_vn'].upper()}")
        print(f"    - Khoảng cách mét: {wall_res['distance_m']}m")
        print(f"    - Độ tin cậy (Confidence): {wall_res['confidence'] * 100:.1f}%")
        print(f"    - Cảnh báo: \"{wall_res['warning']}\"")
        print(f"    - Bounding Box: {wall_res['bounding_box']}")
    else:
        print("    - Kết quả: ❌ Không phát hiện được tường")

    # 2. TEST PHÁT HIỆN BẬC TAM CẤP / CẦU THANG (STAIRS)
    stair_frame, stair_dmap = create_synthetic_stairs()
    t0 = time.time()
    stair_res = geometric_hazard_detector.detect_stairs_and_curbs(stair_frame, stair_dmap)
    t_stair_ms = (time.time() - t0) * 1000

    print(f"\n[2] TEST BẬC TAM CẤP / CẦU THANG:")
    print(f"    - Thời gian tính toán: {t_stair_ms:.2f} ms")
    if stair_res:
        print(f"    - Kết quả: ✅ PHÁT HIỆN {stair_res['class_name_vn'].upper()}")
        print(f"    - Số bậc phát hiện được: {stair_res.get('num_steps', 0)} bậc")
        print(f"    - Khoảng cách mét: {stair_res['distance_m']}m")
        print(f"    - Cảnh báo: \"{stair_res['warning']}\"")
        print(f"    - Bounding Box: {stair_res['bounding_box']}")
    else:
        print("    - Kết quả: ❌ Không phát hiện được bậc thang")

    # 3. TEST TRÊN ẢNH THẬT NẾU CÓ DEPTH MAP
    sample_img_path = os.path.join(ROOT, "test_images", "sample.jpg")
    if os.path.exists(sample_img_path):
        real_img = cv2.imread(sample_img_path)
        # Giả lập depth map dốc đều mặt đường
        h, w = real_img.shape[:2]
        road_dmap = np.linspace(0.8, 6.0, h, dtype=np.float32)[:, None] * np.ones((1, w), dtype=np.float32)
        wall_test = geometric_hazard_detector.detect_wall(road_dmap)
        print(f"\n[3] TEST TRÊN MẶT ĐƯỜNG PHẲNG BÌNH THƯỜNG (Tránh báo động giả - False Positive):")
        if wall_test is None:
            print(f"    - Kết quả: ✅ CHÍNH XÁC (Mặt đường thông thoáng, không báo nhầm tường)")
        else:
            print(f"    - Kết quả: ⚠️ Báo nhầm: {wall_test['class_name_vn']}")

    print("\n==================================================")
    print("✅ HOÀN THÀNH KIỂM THỬ THUẬT TOÁN HÌNH HỌC!")
    print("==================================================")


if __name__ == "__main__":
    test_geometric_detector()
