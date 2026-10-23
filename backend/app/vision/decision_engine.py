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
    "bicycle": "chiếc xe đạp",
    "car": "chiếc xe ô tô",
    "motorcycle": "chiếc xe máy",
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
    "bird": "con chim",
    "cat": "con mèo",
    "dog": "con chó",
    "horse": "con ngựa",
    "sheep": "con cừu",
    "cow": "con bò",
    "elephant": "con voi",
    "bear": "con gấu",
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
    "wine glass": "ly rượu",
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
    "hot dog": "bánh mì kẹp xúc xích",
    "pizza": "bánh pizza",
    "donut": "bánh donut",
    "cake": "bánh ngọt",
    "chair": "chiếc ghế",
    "couch": "ghế sofa",
    "potted plant": "chậu cây",
    "bed": "giường ngủ",
    "dining table": "bàn ăn",
    "table": "chiếc bàn",
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
    "stairs": "cầu thang"
}

def get_vietnamese_name(class_name: str) -> str:
    """Trả về tên tiếng Việt của đối tượng, nếu không có thì trả về class_name gốc."""
    if not class_name:
        return "vật cản"
    return VOCAB.get(str(class_name).lower().strip(), str(class_name))


def generate_guidance(detections: list) -> dict:
    """
    Nhận danh sách các vật thể, lọc ra vật thể quan trọng và ghép thành câu tiếng Việt tự nhiên.
    """
    if not detections:
        return None
        
    parts = []
    has_person = False
    for det in detections:
        conf = getattr(det, "confidence", 0.0)
        # Bỏ qua các nhận diện có độ tin cậy thấp (< 0.45)
        if conf < 0.45:
            continue
            
        cname = getattr(det, "class_name", "")
        # Nếu chưa có trong vocab dùng tên gốc (đảm bảo test test_unknown_class_uses_english_name pass)
        name_vn = VOCAB.get(cname.lower().strip(), cname) if cname else "vật cản"
        if cname == "person":
            has_person = True
            
        # Ước lượng khoảng cách từ bounding box
        dist_text = ""
        bbox = getattr(det, 'bounding_box', None)
        if bbox:
            y1 = getattr(bbox, 'y1', 0)
            y2 = getattr(bbox, 'y2', 0)
            h = y2 - y1
            if h > 300:
                dist_text = " khoảng cách rất gần"
            elif h > 150:
                dist_text = " khoảng 1 đến 2 mét"
            else:
                dist_text = " khoảng 3 mét"
        
        pos = getattr(det, "position", "center")
        if pos == "left":
            parts.append(f"bên trái có {name_vn}{dist_text}")
        elif pos == "right":
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

