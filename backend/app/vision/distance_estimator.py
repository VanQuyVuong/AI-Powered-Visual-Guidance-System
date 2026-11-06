"""
Module ước lượng khoảng cách camera lỗ kim (Pinhole Camera Distance Estimator)
Dành cho hệ thống hỗ trợ di chuyển người khiếm thị.
"""

# Thông số camera giả định (điện thoại cầm ngang ngực, FOV dọc ~50°, FOV ngang ~60°)
CAMERA_HEIGHT_M = 1.4
PERSON_HEIGHT_M = 1.7
PERSON_SHOULDER_WIDTH_M = 0.45
MAX_DISTANCE_M = 15.0

# Chiều cao thực tế trung bình (mét) của các vật thể phổ biến để đo khoảng cách chuẩn ngay cả khi đặt trên bàn
OBJECT_REAL_HEIGHTS_M = {
    "bottle": 0.24,
    "wine glass": 0.18,
    "cup": 0.12,
    "bowl": 0.10,
    "cell phone": 0.15,
    "book": 0.22,
    "laptop": 0.25,
    "keyboard": 0.14,
    "mouse": 0.08,
    "remote": 0.16,
    "vase": 0.25,
    "clock": 0.25,
    "potted plant": 0.45,
    "backpack": 0.45,
    "handbag": 0.35,
    "suitcase": 0.65,
    "chair": 0.85,
    "couch": 0.85,
    "table": 0.75,
    "dining table": 0.75,
    "bed": 0.70,
    "bench": 0.80,
    "tv": 0.55,
    "refrigerator": 1.65,
    "sink": 0.85,
    "door": 2.0,
    "cabinet": 1.5,
    "car": 1.5,
    "motorcycle": 1.1,
    "bicycle": 1.0,
    "bus": 3.0,
    "truck": 2.8,
    "dog": 0.5,
    "cat": 0.3,
}

# Các đồ vật nhỏ thường đặt trên mặt bàn/kệ (không chạm sàn nhà)
TABLETOP_CLASSES = {
    "bottle", "wine glass", "cup", "bowl", "cell phone", "book",
    "laptop", "keyboard", "mouse", "remote", "vase", "clock"
}


def estimate_distance_m(cname: str, box: dict, img_w: int, img_h: int) -> float:
    """
    Ước lượng khoảng cách (mét) bằng hình học camera lỗ kim (pinhole):
    - Người: d = f * H_thật / h_pixel (dùng chiều cao; nếu bị cắt khỏi khung thì dùng bề rộng vai)
    - Đồ vật đặt trên bàn (chai nước, cốc, laptop...): d = f * H_thật / h_pixel
    - Vật chạm đất: kết hợp chiều cao thật và điểm chạm đất (y2)
    """
    box_h = max(1.0, float(box["y2"] - box["y1"]))
    box_w = max(1.0, float(box["x2"] - box["x1"]))
    f_y = (img_h / 2.0) / 0.4663   # tan(25°)
    f_x = (img_w / 2.0) / 0.5774   # tan(30°)

    touches_bottom = box["y2"] >= img_h * 0.95
    touches_top = box["y1"] <= img_h * 0.05
    c_lower = (cname or "").lower().strip()

    if c_lower == "person":
        d_h = f_y * PERSON_HEIGHT_M / box_h
        d_w = f_x * PERSON_SHOULDER_WIDTH_M / box_w
        dist = d_w if (touches_bottom or touches_top) else d_h
    elif c_lower in TABLETOP_CLASSES:
        real_h = OBJECT_REAL_HEIGHTS_M.get(c_lower, 0.22)
        dist = f_y * real_h / box_h
    else:
        horizon_y = img_h * 0.35
        below = float(box["y2"]) - horizon_y
        if below <= 1.0:
            dist = max(0.8, 2.5 * (img_h / max(box_h, 1.0)) * 0.25)
        else:
            dist = f_y * CAMERA_HEIGHT_M / below

        if c_lower in OBJECT_REAL_HEIGHTS_M:
            d_obj = f_y * OBJECT_REAL_HEIGHTS_M[c_lower] / box_h
            dist = min(dist, d_obj)

        if box_h / img_h > 0.45:
            dist = min(dist, 1.2)
        elif box_h / img_h > 0.25:
            dist = min(dist, 2.0)

    return round(min(MAX_DISTANCE_M, max(0.3, dist)), 1)
