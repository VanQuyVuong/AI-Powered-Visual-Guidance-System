"""
Module ước lượng khoảng cách camera lỗ kim (Pinhole Camera Distance Estimator)
Dành cho hệ thống hỗ trợ di chuyển người khiếm thị.
"""

# Thông số camera giả định (điện thoại cầm ngang ngực, FOV dọc ~50°, FOV ngang ~60°)
CAMERA_HEIGHT_M = 1.4
PERSON_HEIGHT_M = 1.7
PERSON_SHOULDER_WIDTH_M = 0.45
MAX_DISTANCE_M = 15.0


def estimate_distance_m(cname: str, box: dict, img_w: int, img_h: int) -> float:
    """
    Ước lượng khoảng cách (mét) bằng hình học camera lỗ kim (pinhole):
    - Người: d = f * H_thật / h_pixel (dùng chiều cao; nếu bị cắt khỏi khung thì dùng bề rộng vai)
    - Vật khác: điểm chạm đất (y2) hoặc kích thước khung hình
    """
    box_h = max(1.0, float(box["y2"] - box["y1"]))
    box_w = max(1.0, float(box["x2"] - box["x1"]))
    f_y = (img_h / 2.0) / 0.4663   # tan(25°)
    f_x = (img_w / 2.0) / 0.5774   # tan(30°)

    touches_bottom = box["y2"] >= img_h * 0.95
    touches_top = box["y1"] <= img_h * 0.05

    if cname == "person":
        d_h = f_y * PERSON_HEIGHT_M / box_h
        d_w = f_x * PERSON_SHOULDER_WIDTH_M / box_w
        dist = d_w if (touches_bottom or touches_top) else d_h
    else:
        horizon_y = img_h * 0.35
        below = float(box["y2"]) - horizon_y
        if below <= 1.0:
            dist = max(0.8, 2.5 * (img_h / max(box_h, 1.0)) * 0.25)
        else:
            dist = f_y * CAMERA_HEIGHT_M / below
        if box_h / img_h > 0.45:
            dist = min(dist, 1.2)
        elif box_h / img_h > 0.25:
            dist = min(dist, 2.0)

    return round(min(MAX_DISTANCE_M, max(0.3, dist)), 1)
