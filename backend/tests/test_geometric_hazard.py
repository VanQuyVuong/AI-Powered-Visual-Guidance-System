import pytest
import numpy as np
from app.vision.geometric_hazard_detector import GeometricHazardDetector


@pytest.fixture
def detector():
    return GeometricHazardDetector()


def test_detect_wall_on_flat_vertical_surface(detector):
    """Bức tường phẳng thẳng đứng ở vùng trung tâm (dZ/dy ~ 0) phải được phát hiện."""
    h, w = 480, 640
    dmap = np.zeros((h, w), dtype=np.float32)

    # Nửa dưới là sàn nhà dốc lên
    for y in range(int(h * 0.65), h):
        ratio = (h - y) / (h * 0.35)
        dmap[y, :] = 0.5 + (1.0 - ratio) * 0.9

    # Nửa trên là bức tường phẳng 1.3m
    dmap[:int(h * 0.65), :] = 1.3

    res = detector.detect_wall(dmap)
    assert res is not None
    assert res["class_name"] == "wall"
    assert res["class_name_vn"] == "bức tường"
    assert 1.1 <= res["distance_m"] <= 1.5
    assert res["is_danger"] is True


def test_detect_wall_ignores_open_road(detector):
    """Mặt đường phẳng thông thoáng (dZ/dy tăng đều đặn) không được báo nhầm là tường."""
    h, w = 480, 640
    # Đường phẳng dốc đều từ 0.8m lên 8.0m
    dmap = np.linspace(0.8, 8.0, h, dtype=np.float32)[:, None] * np.ones((1, w), dtype=np.float32)

    res = detector.detect_wall(dmap)
    assert res is None, "Mặt đường thông thoáng không được báo nhầm là bức tường"


def test_detect_stairs_horizontal_edges_with_depth_jumps(detector):
    """Bậc thang với các vệt ngang và bước nhảy độ sâu phải được nhận diện."""
    h, w = 480, 640
    import cv2
    frame = np.full((h, w, 3), 190, dtype=np.uint8)
    dmap = np.zeros((h, w), dtype=np.float32)

    # 3 bậc ngang rõ rệt
    step_ys = [int(h * 0.55), int(h * 0.68), int(h * 0.80)]
    for y in step_ys:
        cv2.line(frame, (int(w * 0.15), y), (int(w * 0.85), y), (20, 20, 20), 4)

    # Độ sâu nhảy bậc
    cur_z = 2.4
    prev_y = 0
    for y in step_ys:
        dmap[prev_y:y, :] = cur_z
        cur_z -= 0.4
        prev_y = y
    dmap[prev_y:, :] = 0.8

    res = detector.detect_stairs_and_curbs(frame, dmap)
    assert res is not None
    assert res["class_name"] in ("stairs", "step")
    assert res["distance_m"] > 0
    assert res["is_danger"] is True
