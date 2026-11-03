def determine_position(x1: int, x2: int, image_width: int = 640) -> str:
    """
    Tính toán vị trí của vật thể (Left, Center, Right) dựa vào tọa độ Bounding Box.
    Chia khung hình làm 3 phần bằng nhau.
    """
    # Lấy điểm tâm (ở giữa) của vật thể theo trục X
    center_x = (x1 + x2) / 2
    
    third = image_width / 3
    
    if center_x < third:
        return "left"
    elif center_x > 2 * third:
        return "right"
    else:
        return "center"


def is_in_roi(x1: int, y1: int, x2: int, y2: int, img_width: int, img_height: int) -> bool:
    """
    Thuật toán VÙNG QUAN TÂM (Region of Interest - ROI):
    Hình thang phối cảnh 3D mô phỏng làn đường an toàn phía trước mặt người đeo kính.
    - Đáy trên hẹp ở chân trời (rộng 35% ở giữa, tại y = 45% chiều cao)
    - Đáy dưới mở rộng ở chân người đi (rộng 75% ở giữa, tại y = 100% chiều cao)
    Chỉ cảnh báo khi điểm chân tiếp đất của vật thể (bottom-center) lọt vào hình thang này.
    """
    obj_bottom_x = (x1 + x2) / 2.0
    obj_bottom_y = float(y2)
    
    top_y = img_height * 0.40
    # 1. Trục Y: Nếu vật thể lơ lửng trên cao / móc trên tường -> Bỏ qua, lối đi an toàn
    if obj_bottom_y < top_y:
        return False
        
    center_x = img_width * 0.5
    top_width = img_width * 0.40
    bot_width = img_width * 0.80
    
    # Tỷ lệ tiến từ đỉnh hình thang xuống đáy (0.0 ở top_y, 1.0 ở đáy màn hình)
    progress = min(1.0, max(0.0, (obj_bottom_y - top_y) / (img_height - top_y)))
    cur_half_width = (top_width / 2.0) + progress * ((bot_width - top_width) / 2.0)
    
    # Kiểm tra xem tâm chân vật thể có nằm trong phạm vi bề rộng hình thang tại độ sâu đó không
    return abs(obj_bottom_x - center_x) <= cur_half_width


def estimate_distance(y1: int, y2: int, img_height: int) -> str:
    """
    Ước lượng khoảng cách dựa trên tỷ lệ chiều cao của vật thể so với khung hình.
    Vật càng to (chiếm nhiều tỷ lệ khung hình) -> Càng gần.
    """
    obj_height = y2 - y1
    ratio = obj_height / img_height
    
    if ratio > 0.6:
        return "rất gần"
    elif ratio > 0.3:
        return "ở khoảng cách vừa"
    else:
        return "ở xa"


# Từ điển dịch tên tiếng Anh của YOLO sang tiếng Việt (COCO 80 classes + vật dụng quen thuộc)
VOCAB = {
    "person": "người",
    "bicycle": "xe đạp",
    "car": "xe ô tô",
    "motorcycle": "xe máy",
    "airplane": "máy bay",
    "bus": "xe buýt",
    "train": "tàu hỏa",
    "truck": "xe tải",
    "boat": "thuyền",
    "traffic light": "đèn tín hiệu",
    "fire hydrant": "trụ nước chữa cháy",
    "stop sign": "biển báo dừng",
    "parking meter": "đồng hồ đỗ xe",
    "bench": "ghế dài",
    "bird": "chim",
    "cat": "mèo",
    "dog": "chó",
    "horse": "ngựa",
    "sheep": "cừu",
    "cow": "bò",
    "elephant": "voi",
    "bear": "gấu",
    "zebra": "ngựa vằn",
    "giraffe": "hươu cao cổ",
    "backpack": "ba lô",
    "handbag": "túi xách",
    "tie": "cà vạt",
    "suitcase": "vali",
    "frisbee": "đĩa bay",
    "skis": "ván trượt tuyết",
    "snowboard": "ván trượt",
    "sports ball": "quả bóng",
    "kite": "con diều",
    "baseball bat": "gậy bóng chày",
    "baseball glove": "găng tay bóng chày",
    "skateboard": "ván trượt",
    "surfboard": "ván lướt sóng",
    "tennis racket": "vợt tennis",
    "bottle": "chai nước",
    "wine glass": "ly nước",
    "cup": "cốc nước",
    "fork": "cái nĩa",
    "knife": "con dao",
    "spoon": "cái thìa",
    "bowl": "cái bát",
    "banana": "quả chuối",
    "apple": "quả táo",
    "sandwich": "bánh mì kẹp",
    "orange": "quả cam",
    "broccoli": "bông cải xanh",
    "carrot": "củ cà rốt",
    "hot dog": "bánh mì xúc xích",
    "pizza": "bánh pizza",
    "donut": "bánh donut",
    "cake": "bánh ngọt",
    "chair": "ghế",
    "couch": "ghế sofa",
    "potted plant": "chậu cây",
    "bed": "giường ngủ",
    "dining table": "bàn ăn",
    "table": "bàn",
    "toilet": "bồn cầu",
    "tv": "màn hình tivi",
    "laptop": "máy tính xách tay",
    "mouse": "chuột máy tính",
    "remote": "điều khiển",
    "keyboard": "bàn phím",
    "cell phone": "điện thoại",
    "microwave": "lò vi sóng",
    "oven": "lò nướng",
    "toaster": "máy nướng bánh",
    "sink": "bồn rửa",
    "refrigerator": "tủ lạnh",
    "book": "cuốn sách",
    "clock": "đồng hồ",
    "vase": "bình hoa",
    "scissors": "cái kéo",
    "teddy bear": "gấu bông",
    "hair drier": "máy sấy tóc",
    "toothbrush": "bàn chải",
    "door": "cửa ra vào",
    "stair": "bậc thang",
    "stairs": "cầu thang",
    "wall": "bức tường",
    "column": "cột nhà",
    "pillar": "cột nhà",
    "cabinet": "tủ",
    "step": "bậc tam cấp",
    "curb": "bậc tam cấp",
    "obstacle": "vật cản",
    "pothole": "ổ gà"
}

def get_vietnamese_name(class_name: str) -> str:
    """Trả về tên tiếng Việt của đối tượng, nếu không có thì trả về class_name gốc."""
    if not class_name:
        return "vật cản"
    return VOCAB.get(str(class_name).lower().strip(), str(class_name))


def generate_guidance(detections: list) -> dict:
    """
    Nhận danh sách các vật thể, chọn lọc vật quan trọng nhất và sinh câu cảnh báo
    siêu ngắn gọn, phản xạ tức thì (ví dụ: 'Phía trước 2 mét có ghế.').
    """
    if not detections:
        return None
        
    parts = []
    has_person = False
    min_dist_found = 99.0
    
    # Sắp xếp ưu tiên: center trước mặt lên đầu, vật ở gần lên đầu
    def _sort_key(d):
        pos = getattr(d, "position", "center")
        pos_score = 0 if pos == "center" else 1
        dist = getattr(d, "distance_m", None) or getattr(d, "dist_m", None) or 5.0
        return (pos_score, dist)
        
    sorted_dets = sorted(detections, key=_sort_key)

    # Kiểm tra trường hợp đặc biệt: Có vật cản chắn ngang cửa ra vào
    door_det = next((d for d in sorted_dets if getattr(d, "class_name", "") == "door" and getattr(d, "confidence", 0.0) >= 0.40), None)
    if door_det:
        door_dist = getattr(door_det, "distance_m", None) or getattr(door_det, "dist_m", None) or 3.0
        door_pos = getattr(door_det, "position", "center")
        for d in sorted_dets:
            c = getattr(d, "class_name", "")
            if c and c != "door" and getattr(d, "confidence", 0.0) >= 0.40:
                obs_dist = getattr(d, "distance_m", None) or getattr(d, "dist_m", None) or 2.0
                obs_pos = getattr(d, "position", "center")
                if (obs_pos == door_pos or obs_pos == "center") and obs_dist <= door_dist:
                    obs_name = VOCAB.get(c.lower().strip(), c)
                    d_txt = f"{round(obs_dist)} mét " if obs_dist and obs_dist > 0 else ""
                    return {
                        "text": f"Phía trước {d_txt}có {obs_name} chắn ngang cửa ra vào.".replace("  ", " "),
                        "has_person": c == "person",
                        "priority": "HIGH"
                    }

    for det in sorted_dets:
        conf = getattr(det, "confidence", 0.0)
        # Bỏ qua các nhận diện có độ tin cậy thấp (< 0.40) theo chuẩn Unit Test
        if conf < 0.40:
            continue
            
        cname = getattr(det, "class_name", "")
        name_vn = VOCAB.get(cname.lower().strip(), cname) if cname else "vật cản"
        if cname == "person":
            has_person = True
            
        # Xác định khoảng cách mét
        dist_val = getattr(det, "distance_m", None) or getattr(det, "dist_m", None)
        dist_text = ""
        if dist_val is not None and dist_val > 0:
            if dist_val <= 1.0:
                dist_text = "1 mét "
            elif dist_val <= 1.6:
                dist_text = "1 mét rưỡi "
            else:
                dist_text = f"{round(dist_val)} mét "
        else:
            bbox = getattr(det, 'bounding_box', None)
            if bbox:
                y1 = getattr(bbox, 'y1', 0)
                y2 = getattr(bbox, 'y2', 0)
                h = y2 - y1
                if h > 300:
                    dist_text = "rất gần "
                elif h > 150:
                    dist_text = "1 đến 2 mét "
                else:
                    dist_text = "3 mét "
        
        pos = getattr(det, "position", "center")
        if pos == "left":
            parts.append(f"bên trái {dist_text}có {name_vn}".replace("  ", " "))
        elif pos == "right":
            parts.append(f"bên phải {dist_text}có {name_vn}".replace("  ", " "))
        else:
            parts.append(f"Phía trước {dist_text}có {name_vn}".replace("  ", " "))
            
    if not parts:
        return None
        
    # Ghép câu ngắn gọn: tối đa 2 vật quan trọng nhất
    if len(parts) == 1:
        text = parts[0].strip().capitalize() + "."
    else:
        text = parts[0].strip().capitalize() + " và " + parts[1].strip() + "."
    
    return {
        "text": text,
        "has_person": has_person,
        "priority": "HIGH" if "phía trước" in text.lower() or has_person else "MEDIUM"
    }


# ─── 3-LANE NAVIGATION ENGINE (ĐIỀU HƯỚNG 3 LÀN THỜI GIAN THỰC) ───
SYSTEM_MODE = "NAVIGATION"  # NAVIGATION (hướng dẫn đường) hoặc CONVERSATION (hỏi đáp)

def set_mode(mode: str):
    """Chuyển đổi chế độ hoạt động"""
    global SYSTEM_MODE
    if mode in ["NAVIGATION", "CONVERSATION"]:
        SYSTEM_MODE = mode

def get_mode() -> str:
    """Lấy chế độ hiện tại"""
    return SYSTEM_MODE

def generate_navigation_direction(detections: list, img_width: int = 640, img_height: int = 480) -> dict:
    """
    Thuật toán điều hướng 3 làn (3-Lane Navigation Engine):
    Phân tích mức độ cản trở ở 3 làn: Trái (Left), Giữa (Center), Phải (Right).
    Trả về:
    {
        "direction": "straight" | "left" | "right" | "stop",
        "confidence": float,
        "text": "Đi thẳng" | "Rẽ trái" | "Rẽ phải" | "Dừng lại",
        "scores": {"center": int, "left": int, "right": int}
    }
    """
    if not detections:
        return {
            "direction": "straight",
            "confidence": 0.95,
            "text": "Đi thẳng",
            "scores": {"center": 0, "left": 0, "right": 0}
        }

    center_score = 0
    left_score = 0
    right_score = 0

    for d in detections:
        pos = getattr(d, "position", None) or (d.get("position") if isinstance(d, dict) else "center")
        dist = getattr(d, "distance_m", None) or (d.get("distance_m") if isinstance(d, dict) else None) or 5.0
        is_dang = getattr(d, "is_danger", None) or (d.get("is_danger") if isinstance(d, dict) else False)

        if dist <= 1.5 or is_dang:
            weight = 3
        elif dist <= 3.0:
            weight = 2
        elif dist <= 4.5:
            weight = 1
        else:
            weight = 0

        if pos == "center":
            center_score += weight
        elif pos == "left":
            left_score += weight
        elif pos == "right":
            right_score += weight

    scores = {"center": center_score, "left": left_score, "right": right_score}

    # Cả 3 hướng đều bị chặn nặng -> DỪNG LẠI
    if center_score >= 2 and left_score >= 2 and right_score >= 2:
        return {"direction": "stop", "confidence": 0.95, "text": "Dừng lại", "scores": scores}

    # Làn giữa có vật cản -> Rẽ sang làn an toàn hơn
    if center_score >= 2:
        if left_score <= right_score:
            return {"direction": "left", "confidence": min(0.9, 0.5 + center_score * 0.1), "text": "Rẽ trái", "scores": scores}
        else:
            return {"direction": "right", "confidence": min(0.9, 0.5 + center_score * 0.1), "text": "Rẽ phải", "scores": scores}
    elif center_score >= 1:
        if left_score <= right_score:
            return {"direction": "left", "confidence": 0.7, "text": "Rẽ trái nhẹ", "scores": scores}
        else:
            return {"direction": "right", "confidence": 0.7, "text": "Rẽ phải nhẹ", "scores": scores}
    else:
        return {"direction": "straight", "confidence": 0.9, "text": "Đi thẳng", "scores": scores}


