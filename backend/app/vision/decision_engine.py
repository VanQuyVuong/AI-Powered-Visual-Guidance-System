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
    "doorstep": "bậc thềm",
    "threshold": "bậc thềm",
    "multi_obstacle": "nhiều vật cản",
    "obstacle": "vật cản",
    "pothole": "ổ gà"
}

SMALL_CLUTTER_CLASSES = {
    "bottle", "wine glass", "cup", "bowl", "cell phone", "book",
    "laptop", "keyboard", "mouse", "remote", "vase", "clock",
    "backpack", "handbag", "scissors", "teddy bear", "toothbrush"
}


def get_vietnamese_name(class_name: str) -> str:
    """Trả về tên tiếng Việt của đối tượng, nếu không có thì trả về class_name gốc."""
    if not class_name:
        return "vật cản"
    return VOCAB.get(str(class_name).lower().strip(), str(class_name))


def _format_dist_text(det) -> str:
    dist_val = getattr(det, "distance_m", None) or getattr(det, "dist_m", None)
    if dist_val is not None and dist_val > 0:
        if dist_val <= 1.0:
            return "1 mét "
        elif dist_val <= 1.6:
            return "1 mét rưỡi "
        else:
            return f"{round(dist_val)} mét "
    bbox = getattr(det, "bounding_box", None)
    if bbox:
        y1 = getattr(bbox, "y1", 0)
        y2 = getattr(bbox, "y2", 0)
        h = y2 - y1
        if h > 300:
            return "rất gần "
        elif h > 150:
            return "1 đến 2 mét "
        else:
            return "3 mét "
    return ""


def generate_guidance(detections: list) -> dict:
    """
    Nhận danh sách các vật thể, GOM NHÓM các vật cản đứng gần nhau để tránh đọc lặp gây ồn,
    và sinh câu cảnh báo ngắn gọn, phản xạ tức thì.
    """
    if not detections:
        return None

    # Lọc các nhận diện có độ tin cậy >= 0.40
    valid_dets = [d for d in detections if getattr(d, "confidence", 0.0) >= 0.40]
    if not valid_dets:
        return None

    has_person = any(getattr(d, "class_name", "").lower() == "person" for d in valid_dets)

    # Sắp xếp ưu tiên: center trước mặt lên đầu, vật ở gần lên đầu
    def _sort_key(d):
        pos = getattr(d, "position", "center")
        pos_score = 0 if pos == "center" else 1
        dist = getattr(d, "distance_m", None) or getattr(d, "dist_m", None) or 5.0
        return (pos_score, dist)

    sorted_dets = sorted(valid_dets, key=_sort_key)

    # 1. Kiểm tra trường hợp đặc biệt: Có vật cản chắn ngang cửa ra vào
    door_det = next((d for d in sorted_dets if getattr(d, "class_name", "") == "door"), None)
    if door_det:
        door_dist = getattr(door_det, "distance_m", None) or getattr(door_det, "dist_m", None) or 3.0
        door_pos = getattr(door_det, "position", "center")
        for d in sorted_dets:
            c = getattr(d, "class_name", "")
            if c and c != "door":
                obs_dist = getattr(d, "distance_m", None) or getattr(d, "dist_m", None) or 2.0
                obs_pos = getattr(d, "position", "center")
                if (obs_pos == door_pos or obs_pos == "center") and obs_dist <= door_dist:
                    obs_name = getattr(d, "class_name_vn", None) or VOCAB.get(c.lower().strip(), c)
                    d_txt = f"{round(obs_dist)} mét " if obs_dist and obs_dist > 0 else ""
                    return {
                        "text": f"Phía trước {d_txt}có {obs_name} chắn ngang cửa ra vào.".replace("  ", " "),
                        "has_person": c == "person" or has_person,
                        "priority": "HIGH"
                    }

    # 2. Ưu tiên cho Bậc thềm / Cầu thang / Ổ gà / Bức tường gần (<= 1.35m)
    def _is_special_hazard(d):
        c = getattr(d, "class_name", "").lower()
        if c in ("stairs", "step", "curb", "doorstep", "threshold", "pothole"):
            return True
        if c == "wall":
            d_val = getattr(d, "distance_m", None) or getattr(d, "dist_m", None) or 5.0
            return d_val <= 1.35
        return False

    special_hazards = [d for d in sorted_dets if _is_special_hazard(d)]
    if special_hazards:
        top_hz = special_hazards[0]
        hz_name = getattr(top_hz, "class_name_vn", None) or VOCAB.get(getattr(top_hz, "class_name", "").lower(), "vật cản")
        dist_text = _format_dist_text(top_hz)
        pos = getattr(top_hz, "position", "center")
        prefix = "bên trái" if pos == "left" else ("bên phải" if pos == "right" else "Phía trước")
        text = f"{prefix} {dist_text}có {hz_name}.".replace("  ", " ").strip()
        text = text[0].upper() + text[1:]
        return {
            "text": text,
            "has_person": has_person,
            "priority": "HIGH"
        }

    # 3. Kiểm tra nếu có Bàn (table/dining table) kèm nhiều đồ vật nhỏ trên bàn (chai nước, cốc...)
    has_table = any(getattr(d, "class_name", "").lower() in ("table", "dining table", "desk") for d in sorted_dets)
    clutter_dets = [d for d in sorted_dets if getattr(d, "class_name", "").lower() in SMALL_CLUTTER_CLASSES]

    def _describe_cluster(objs: list, on_table: bool = False) -> str:
        """Nếu cụm toàn vật cùng loại (VD: nhiều chai nước) -> 'nhiều chai nước'; nếu khác loại -> 'nhiều vật cản'."""
        names_vn = []
        for o in objs:
            cn = getattr(o, "class_name", "").lower().strip()
            if cn in ("table", "dining table", "desk"):
                continue
            vn = getattr(o, "class_name_vn", None) or VOCAB.get(cn, cn or "vật cản")
            names_vn.append(vn)
        if not names_vn:
            return "bàn" if on_table else "nhiều vật cản"
        # Đếm xem có loại nào chiếm đa số (>= 2 vật cùng tên) không
        from collections import Counter
        top_name, count = Counter(names_vn).most_common(1)[0]
        if len(set(names_vn)) == 1 or count >= 2:
            if on_table:
                return f"nhiều {top_name} trên bàn"
            return f"nhiều {top_name}"
        return "nhiều đồ vật trên bàn" if on_table else "nhiều vật cản"

    if has_table and len(clutter_dets) >= 2:
        table_det = next(d for d in sorted_dets if getattr(d, "class_name", "").lower() in ("table", "dining table", "desk"))
        dist_text = _format_dist_text(table_det)
        pos = getattr(table_det, "position", "center")
        prefix = "bên trái" if pos == "left" else ("bên phải" if pos == "right" else "Phía trước")
        cluster_desc = _describe_cluster(clutter_dets, on_table=True)
        text = f"{prefix} {dist_text}có {cluster_desc}.".replace("  ", " ").strip()
        text = text[0].upper() + text[1:]
        return {
            "text": text,
            "has_person": has_person,
            "priority": "HIGH" if pos == "center" or has_person else "MEDIUM"
        }
    elif has_table and len(clutter_dets) == 1:
        sorted_dets = [d for d in sorted_dets if getattr(d, "class_name", "").lower() not in SMALL_CLUTTER_CLASSES]

    # 4. Nếu có từ 3 vật cản trở lên đứng gần nhau -> Gom thành 1 cụm ("nhiều <tên vật>" hoặc "nhiều vật cản")
    non_person_dets = [d for d in sorted_dets if getattr(d, "class_name", "").lower() != "person"]
    if len(non_person_dets) >= 3:
        nearest_d = non_person_dets[0]
        dist_text = _format_dist_text(nearest_d)
        has_center = any(getattr(d, "position", "center") == "center" for d in non_person_dets)
        pos = "center" if has_center else getattr(nearest_d, "position", "center")
        prefix = "bên trái" if pos == "left" else ("bên phải" if pos == "right" else "Phía trước")
        cluster_desc = _describe_cluster(non_person_dets, on_table=has_table)
        text = f"{prefix} {dist_text}có {cluster_desc}.".replace("  ", " ").strip()
        text = text[0].upper() + text[1:]
        return {
            "text": text,
            "has_person": has_person,
            "priority": "HIGH" if pos == "center" or has_person else "MEDIUM"
        }

    # 5. Gom nhóm theo từng hướng (center, left, right): Nếu cùng 1 hướng có >= 2 vật cản -> gộp lại
    grouped_by_pos = {"center": [], "left": [], "right": []}
    for d in sorted_dets:
        p = getattr(d, "position", "center")
        grouped_by_pos.setdefault(p, []).append(d)

    parts = []
    for pos in ("center", "left", "right"):
        group = grouped_by_pos.get(pos, [])
        if not group:
            continue

        prefix = "bên trái" if pos == "left" else ("bên phải" if pos == "right" else "Phía trước")
        first_det = group[0]
        dist_text = _format_dist_text(first_det)

        if len(group) >= 2:
            person_in_group = next((d for d in group if getattr(d, "class_name", "").lower() == "person"), None)
            if person_in_group:
                p_dist_text = _format_dist_text(person_in_group)
                phrase = f"{prefix} {p_dist_text}có người và vật cản".replace("  ", " ").strip()
            else:
                cluster_desc = _describe_cluster(group, on_table=False)
                phrase = f"{prefix} {dist_text}có {cluster_desc}".replace("  ", " ").strip()
        else:
            cname = getattr(first_det, "class_name", "")
            name_vn = getattr(first_det, "class_name_vn", None) or (VOCAB.get(cname.lower().strip(), cname) if cname else "vật cản")
            phrase = f"{prefix} {dist_text}có {name_vn}".replace("  ", " ").strip()

        if phrase not in parts:
            parts.append(phrase)

    if not parts:
        return None

    # Ghép câu ngắn gọn: tối đa 2 hướng quan trọng nhất
    if len(parts) == 1:
        text = parts[0].strip()
        text = text[0].upper() + text[1:] + "."
    else:
        p0 = parts[0].strip()
        p0 = p0[0].upper() + p0[1:]
        text = p0 + " và " + parts[1].strip() + "."

    return {
        "text": text,
        "has_person": has_person,
        "priority": "HIGH" if "phía trước" in text.lower() or has_person else "MEDIUM"
    }


def build_navigation_speech(
    base_guidance: dict,
    all_objects: list,
    nav_data: dict,
    img_width: int = 640,
    depth_lanes: dict = None,
) -> dict:
    """
    Hàm hợp nhất [Cảnh báo vật cản đã gom cụm] + [Hành động điều hướng 3 làn từ Depth + YOLO]:
    - Làn giữa trống (>= 1.5m) -> "Phía trước an toàn, đi thẳng."
    - Bị chắn giữa, còn 1 bên trống -> "Phía trước có ... Bên trái/phải trống, hãy rẽ trái/phải."
    - Cả 3 làn tắc hoặc sát tường -> "Dừng lại! Phía trước có ... Hãy nhìn sang trái hoặc sang phải để tìm hướng trống."
    - Bậc / thềm dưới chân -> "Chú ý dưới chân, ... hãy bước chậm."
    """
    scores = dict((nav_data or {}).get("scores", {"center": 0, "left": 0, "right": 0}))
    left_score = scores.get("left", 0)
    right_score = scores.get("right", 0)
    center_score = scores.get("center", 0)

    # Kết hợp trực tiếp thông tin 3 làn từ Depth Map (nếu có)
    lanes_map = (depth_lanes or {}).get("lanes", {}) if depth_lanes else {}
    if lanes_map:
        if lanes_map.get("left", {}).get("is_blocked"):
            left_score += 2
        if lanes_map.get("center", {}).get("is_blocked"):
            center_score += 2
        if lanes_map.get("right", {}).get("is_blocked"):
            right_score += 2

    # Kiểm tra cả trong all_objects để biết bên trái/phải có thực sự trống không
    for obj in (all_objects or []):
        p = obj.get("position", "center") if isinstance(obj, dict) else getattr(obj, "position", "center")
        d = (obj.get("distance_m") if isinstance(obj, dict) else getattr(obj, "distance_m", None)) or 4.0
        if d <= 3.0:
            if p == "left":
                left_score += 1
            elif p == "right":
                right_score += 1

    if not base_guidance or not base_guidance.get("text"):
        # Nếu Depth Map báo làn giữa đang bị bịt kín (< 1.35m) dù không có hộp YOLO nào -> CẤM báo an toàn!
        if center_score >= 2:
            if left_score < right_score:
                return {"text": "Cảnh báo! Phía trước có vật cản gần, bên trái trống, hãy rẽ trái.", "has_person": False, "priority": "HIGH"}
            if right_score < left_score:
                return {"text": "Cảnh báo! Phía trước có vật cản gần, bên phải trống, hãy rẽ phải.", "has_person": False, "priority": "HIGH"}
            return {"text": "Dừng lại! Phía trước có vật cản chắn lối, hãy nhìn sang trái hoặc sang phải để tìm hướng trống.", "has_person": False, "priority": "HIGH"}

        return {
            "text": "Phía trước an toàn, đi thẳng.",
            "has_person": False,
            "priority": "LOW"
        }

    import re
    raw_text = base_guidance["text"].strip().rstrip(".")
    priority = base_guidance.get("priority", "HIGH")
    has_person = base_guidance.get("has_person", False)

    # Loại bỏ các cụm số mét nhảy liên tục ("khoảng 2 mét", "cách 1.8 mét") khỏi câu đọc TTS
    # để câu nói ngắn gọn, nghe ngay được hướng đi và không bị đọc lặp khi khoảng cách dao động!
    clean_text = re.sub(r"\s*(khoảng|cách)\s+\d+([.,]\d+)?\s*mét\s*", " ", raw_text, flags=re.IGNORECASE)
    clean_text = re.sub(r"\s+", " ", clean_text).strip().rstrip(".")
    lower_text = clean_text.lower()

    # 1. Trường hợp BỨC TƯỜNG chắn trước mặt
    if "bức tường" in lower_text and len(all_objects or []) < 2:
        if left_score == 0 and right_score > 0:
            action = "hãy rẽ trái."
        elif right_score == 0 and left_score > 0:
            action = "hãy rẽ phải."
        else:
            action = "hãy nhìn sang trái hoặc sang phải."
        return {
            "text": f"Phía trước có bức tường, {action}",
            "has_person": has_person,
            "priority": "HIGH"
        }

    # 2. Trường hợp BẬC THỀM / BẬC TAM CẤP / CẦU THANG dưới chân
    if any(k in lower_text for k in ("bậc thềm", "bậc tam cấp", "cầu thang", "bậc thang")):
        return {
            "text": f"Chú ý dưới chân, {clean_text[0].lower() + clean_text[1:]}, hãy bước chậm.",
            "has_person": has_person,
            "priority": "HIGH"
        }

    # 3. Trường hợp có NHIỀU VẬT CẢN (>= 2 vật thể hoặc câu có chữ "nhiều")
    is_multi_obstacles = (len(all_objects or []) >= 2) or ("nhiều" in lower_text)

    center_objs = [
        o for o in (all_objects or [])
        if (o.get("position") if isinstance(o, dict) else getattr(o, "position", "")) == "center"
    ]
    left_objs = [
        o for o in (all_objects or [])
        if (o.get("position") if isinstance(o, dict) else getattr(o, "position", "")) == "left"
    ]
    right_objs = [
        o for o in (all_objects or [])
        if (o.get("position") if isinstance(o, dict) else getattr(o, "position", "")) == "right"
    ]

    # Nếu các vật cản chỉ nằm lệch hẳn sang BÊN TRÁI hoặc BÊN PHẢI (làn giữa trống):
    if not center_objs and center_score == 0 and "phía trước" not in lower_text:
        if left_objs and not right_objs:
            return {
                "text": f"{clean_text}, bên phải trống, hãy né sang phải và đi thẳng.",
                "has_person": has_person,
                "priority": priority,
            }
        if right_objs and not left_objs:
            return {
                "text": f"{clean_text}, bên trái trống, hãy né sang trái và đi thẳng.",
                "has_person": has_person,
                "priority": priority,
            }

    # Xác định hành động điều hướng ngắn gọn, tự nhiên như người thật nhắc
    if left_score >= 2 and right_score >= 2:
        action = "hãy nhìn sang trái hoặc sang phải."
    elif left_score < right_score:
        action = "bên trái trống, hãy rẽ trái."
    elif right_score < left_score:
        action = "bên phải trống, hãy rẽ phải."
    else:
        left_med = lanes_map.get("left", {}).get("median_m", 0.0) if lanes_map else 0.0
        right_med = lanes_map.get("right", {}).get("median_m", 0.0) if lanes_map else 0.0
        if abs(left_med - right_med) >= 0.25:
            action = "bên trái trống, hãy rẽ trái." if left_med > right_med else "bên phải trống, hãy rẽ phải."
        else:
            if center_objs and not is_multi_obstacles:
                def _get_cx(o):
                    if isinstance(o, dict):
                        if "bounding_box" in o and isinstance(o["bounding_box"], dict):
                            return (o["bounding_box"]["x1"] + o["bounding_box"]["x2"]) / 2.0
                        if "bbox" in o and len(o["bbox"]) >= 4:
                            return (o["bbox"][0] + o["bbox"][2]) / 2.0
                        return img_width * 0.5
                    return (o.bounding_box.x1 + o.bounding_box.x2) / 2.0

                avg_cx = sum(_get_cx(o) for o in center_objs) / len(center_objs)
                if avg_cx >= img_width * 0.5:
                    action = "bên trái trống, hãy rẽ trái."
                else:
                    action = "bên phải trống, hãy rẽ phải."
            else:
                action = "hãy nhìn sang trái hoặc sang phải."

    if is_multi_obstacles:
        return {
            "text": f"Phía trước có nhiều vật cản, {action}",
            "has_person": has_person,
            "priority": "HIGH"
        }

    # 4. Trường hợp chỉ có 1 vật cản chắn ở phía trước (Center)
    if "phía trước" in lower_text or center_score > 0:
        return {
            "text": f"{clean_text}, {action}",
            "has_person": has_person,
            "priority": "HIGH"
        }

    # 5. Trường hợp 1 vật cản chỉ nằm ở BÊN TRÁI hoặc BÊN PHẢI, còn phía trước trống
    return {
        "text": f"{clean_text}, phía trước trống, đi thẳng.",
        "has_person": has_person,
        "priority": priority
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



