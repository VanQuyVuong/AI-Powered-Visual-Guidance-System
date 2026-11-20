"""
Module ước lượng & hiệu chuẩn khoảng cách camera (Pinhole + Multi-Source Calibration)
Dành cho hệ thống hỗ trợ di chuyển người khiếm thị.
"""

from typing import Optional

# Thông số camera giả định (điện thoại cầm ngang ngực, cao ~1.35m)
CAMERA_HEIGHT_M = 1.35
PERSON_HEIGHT_M = 1.70
PERSON_SHOULDER_WIDTH_M = 0.45
MAX_DISTANCE_M = 15.0

# tan(half_fov_y): ảnh dọc điện thoại 3:4 có FOV dọc rộng ~64° (tan 32° ≈ 0.625);
# ảnh ngang 4:3 hoặc webcam thường có FOV dọc ~50°–53° (tan 25.5° ≈ 0.477).
TAN_HALF_FOV_Y_PORTRAIT = 0.625
TAN_HALF_FOV_Y_LANDSCAPE = 0.477
TAN_HALF_FOV_X_PORTRAIT = 0.477
TAN_HALF_FOV_X_LANDSCAPE = 0.600

# Chiều cao thực tế trung bình (mét) của các vật thể phổ biến
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
    "desk": 0.75,
    "bed": 0.70,
    "bench": 0.80,
    "tv": 0.55,
    "refrigerator": 1.65,
    "sink": 0.85,
    "door": 2.00,
    "cabinet": 1.50,
    "car": 1.50,
    "motorcycle": 1.10,
    "bicycle": 1.00,
    "bus": 3.00,
    "truck": 2.80,
    "dog": 0.50,
    "cat": 0.30,
}

# Các đồ vật nhỏ thường đặt trên mặt bàn/kệ (không chạm sàn nhà)
TABLETOP_CLASSES = {
    "bottle", "wine glass", "cup", "bowl", "cell phone", "book",
    "laptop", "keyboard", "mouse", "remote", "vase", "clock",
}


def estimate_distance_m(cname: str, box: dict, img_w: int, img_h: int) -> float:
    """
    Ước lượng khoảng cách (mét) bằng hình học camera lỗ kim (pinhole) đã hiệu chuẩn theo khung dọc/ngang:
    - Tự phân biệt camera dọc điện thoại (FOV dọc ~64°) vs khung ngang (~51°) để không bóp ngắn vật ở xa.
    - Với vật đứng có kích thước chuẩn trong `OBJECT_REAL_HEIGHTS_M` và không bị cắt mép ảnh,
      ưu tiên kết hợp cân bằng kích thước thật `d_obj` và điểm chạm đất `d_ground` thay vì ép trần cứng theo tỷ lệ box.
    """
    w = max(1, int(img_w))
    h = max(1, int(img_h))
    box_h = max(1.0, float(box["y2"] - box["y1"]))
    box_w = max(1.0, float(box["x2"] - box["x1"]))

    is_portrait = h > w
    tan_y = TAN_HALF_FOV_Y_PORTRAIT if is_portrait else TAN_HALF_FOV_Y_LANDSCAPE
    tan_x = TAN_HALF_FOV_X_PORTRAIT if is_portrait else TAN_HALF_FOV_X_LANDSCAPE
    f_y = (h / 2.0) / tan_y
    f_x = (w / 2.0) / tan_x

    touches_bottom = box["y2"] >= h * 0.95
    touches_top = box["y1"] <= h * 0.05
    c_lower = (cname or "").lower().strip()

    if c_lower == "person":
        d_h = f_y * PERSON_HEIGHT_M / box_h
        d_w = f_x * PERSON_SHOULDER_WIDTH_M / box_w
        dist = d_w if (touches_bottom or touches_top) else d_h
    elif c_lower in TABLETOP_CLASSES:
        real_h = OBJECT_REAL_HEIGHTS_M.get(c_lower, 0.22)
        dist = f_y * real_h / box_h
    else:
        horizon_y = h * 0.35
        below = float(box["y2"]) - horizon_y
        if below <= 1.0:
            d_ground = max(1.2, 2.5 * (h / max(box_h, 1.0)) * 0.30)
        else:
            d_ground = f_y * CAMERA_HEIGHT_M / below

        if c_lower in OBJECT_REAL_HEIGHTS_M:
            d_obj = f_y * OBJECT_REAL_HEIGHTS_M[c_lower] / box_h
            if not touches_top and not touches_bottom:
                # Khi thấy trọn vật thể trong khung, d_obj ổn định hơn ngay cả khi người dùng hơi cúi/ngẩng cam
                dist = 0.70 * d_obj + 0.30 * d_ground
            else:
                dist = min(d_ground, d_obj)
        else:
            dist = d_ground

        # Chỉ áp ngưỡng an toàn sát mặt khi vật thực sự chiếm gần trọn khung hình (>= 65%)
        if (box_h / h) >= 0.65 and touches_bottom:
            dist = min(dist, 1.1)

    return round(min(MAX_DISTANCE_M, max(0.3, dist)), 1)


def resolve_calibrated_distance(
    cname: str,
    metric_dist: Optional[float],
    pinhole_dist: float,
    box: dict,
    img_w: int,
    img_h: int,
) -> float:
    """
    Hòa giải thông minh giữa khoảng cách Metric Depth (Depth V2 / EdgeSAM) và Pinhole:
    - Chiều 1 (Vật ở xa bị Depth kéo tụt lại gần): ưu tiên Pinhole khi vật nhỏ ở >= 2.35m.
    - Chiều 2 (Vật sát mặt bị Depth kéo vọt ra xa do chói đèn / nhìn xuyên kính / nền sau lọt vào box):
      ưu tiên Pinhole & tỷ lệ khung hình khi vật chiếm diện tích lớn hoặc ở gần <= 1.35m.
    """
    w = float(max(1, img_w))
    h = float(max(1, img_h))
    bw = max(1.0, float(box["x2"]) - float(box["x1"]))
    bh = max(1.0, float(box["y2"]) - float(box["y1"]))
    box_h_ratio = bh / h
    area_ratio = (bw * bh) / (w * h)

    if metric_dist is None or not (0.2 <= metric_dist <= MAX_DISTANCE_M):
        return round(pinhole_dist, 1)

    touches_bottom = float(box["y2"]) >= h * 0.94
    touches_top = float(box["y1"]) <= h * 0.05
    c_lower = (cname or "").lower().strip()
    has_known_height = (c_lower == "person") or (c_lower in OBJECT_REAL_HEIGHTS_M)

    # 0. Vật thể trùm gần kín khung hình (bàn, tường, tủ sát mặt) -> không bao giờ được đoán xa > 1.4m
    if area_ratio >= 0.45 or box_h_ratio >= 0.65:
        safe_close = min(metric_dist, max(0.4, min(1.4, pinhole_dist * 1.3)))
        return round(safe_close, 1)

    # 0b. Vật ở rất gần (<= 1.35m theo quang học / hộp lớn) nhưng DepthV2 bị chói đèn hoặc dính nền sau (> 2.0m)
    if (pinhole_dist <= 1.35 or area_ratio >= 0.28 or box_h_ratio >= 0.48) and metric_dist > max(2.0, pinhole_dist * 1.75):
        blended_close = 0.75 * pinhole_dist + 0.25 * min(metric_dist, pinhole_dist * 1.6)
        return round(min(1.6, max(0.3, blended_close)), 1)

    # 1. Vật nhỏ/xa rõ rệt (>= 3.0m theo quang học) nhưng depth map chưa kịp cập nhật -> ưu tiên Pinhole
    if pinhole_dist >= 3.0 and box_h_ratio < 0.22 and metric_dist < pinhole_dist * 0.62:
        return round(pinhole_dist, 1)

    # 2. Vật có chiều cao chuẩn thấy trọn trong khung ở tầm trung-xa (>= 2.35m, box_h < 36%)
    #    nhưng Depth Map bị kéo tụt bởi sàn trước mặt hoặc sai số scale camera (< 75% pinhole)
    if (
        has_known_height
        and not touches_bottom
        and not touches_top
        and box_h_ratio < 0.36
        and pinhole_dist >= 2.35
        and metric_dist < pinhole_dist * 0.75
    ):
        blended = 0.25 * metric_dist + 0.75 * pinhole_dist
        return round(blended, 1)

    return round(metric_dist, 1)
