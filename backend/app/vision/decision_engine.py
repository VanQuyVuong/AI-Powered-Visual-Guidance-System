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
    
    top_y = img_height * 0.45
    # 1. Trục Y: Nếu vật thể lơ lửng trên cao / móc trên tường -> Bỏ qua, lối đi an toàn
    if obj_bottom_y < top_y:
        return False
        
    center_x = img_width * 0.5
    top_width = img_width * 0.35
    bot_width = img_width * 0.75
    
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


def generate_guidance(detections: list) -> dict:
    """
    Nhận danh sách các vật thể, lọc ra vật thể quan trọng và ghép thành câu tiếng Việt.
    """
    if not detections:
        return None
        
    # Từ điển dịch tên tiếng Anh của YOLO sang tiếng Việt (COCO 80 classes)
    vocab = {
        "person": "người",
        "car": "chiếc ô tô",
        "motorcycle": "chiếc xe máy",
        "bicycle": "chiếc xe đạp",
        "bus": "xe buýt",
        "truck": "xe tải",
        "chair": "ghế",
        "couch": "ghế sofa",
        "table": "bàn",
        "dining table": "bàn",
        "bed": "giường",
        "tv": "màn hình",
        "laptop": "máy tính",
        "bottle": "chai nước",
        "cup": "cốc nước",
        "backpack": "ba lô",
        "handbag": "túi xách",
        "suitcase": "vali",
        "cell phone": "điện thoại",
        "potted plant": "chậu cây",
        "dog": "con chó",
        "cat": "con mèo",
        "traffic light": "đèn tín hiệu",
        "stop sign": "biển báo",
        "fire hydrant": "trụ nước",
        "door": "cửa",
        "stair": "bậc thang",
        "stairs": "cầu thang"
    }
    
    parts = []
    has_person = False
    for det in detections:
        if det.confidence < 0.45:
            continue
            
        name_vn = vocab.get(det.class_name, "vật cản")
        if det.class_name == "person":
            has_person = True
            
        # Ước lượng khoảng cách
        dist_text = ""
        if hasattr(det, 'bounding_box') and det.bounding_box:
            h = det.bounding_box.y2 - det.bounding_box.y1
            if h > 300:
                dist_text = " khoảng cách rất gần"
            elif h > 150:
                dist_text = " khoảng 1 đến 2 mét"
            else:
                dist_text = " khoảng 3 mét"
        
        if det.position == "left":
            parts.append(f"bên trái có {name_vn}{dist_text}")
        elif det.position == "right":
            parts.append(f"bên phải có {name_vn}{dist_text}")
        else:
            parts.append(f"ngay phía trước có {name_vn}{dist_text}")
            
    if not parts:
        return None
        
    text = "Cảnh báo, " + " và ".join(parts[:2]) + "."
    
    return {
        "text": text,
        "has_person": has_person,
        "priority": "HIGH" if "phía trước" in text.lower() or has_person else "MEDIUM"
    }
