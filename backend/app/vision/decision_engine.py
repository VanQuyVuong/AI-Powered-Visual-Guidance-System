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


def generate_guidance(detections: list) -> dict:
    """
    Nhận danh sách các vật thể, lọc ra vật thể quan trọng và ghép thành câu tiếng Việt.
    """
    if not detections:
        return None
        
    # Từ điển dịch tên tiếng Anh của YOLO sang tiếng Việt (MVP)
    vocab = {
        "person": "người",
        "car": "chiếc xe ô tô",
        "motorcycle": "chiếc xe máy",
        "bicycle": "chiếc xe đạp",
        "bus": "chiếc xe buýt",
        "truck": "chiếc xe tải"
    }
    
    parts = []
    for det in detections:
        # Chỉ cảnh báo các vật thể mà AI chắc chắn trên 50%
        if det.confidence < 0.5:
            continue
            
        # Lấy tên tiếng Việt, nếu không có trong từ điển thì giữ nguyên tiếng Anh
        name_vn = vocab.get(det.class_name, det.class_name)
        
        # Ghép câu dựa trên vị trí
        if det.position == "left":
            parts.append(f"bên trái có {name_vn}")
        elif det.position == "right":
            parts.append(f"bên phải có {name_vn}")
        else:
            parts.append(f"phía trước có {name_vn}")
            
    # Nếu lọc xong mà không có vật thể nào đủ độ tin cậy
    if not parts:
        return None
        
    # Ghép các vế lại với nhau bằng chữ "và", viết hoa chữ cái đầu và thêm dấu chấm
    text = " và ".join(parts).capitalize() + "."
    
    return {
        "text": text,
        "priority": "HIGH" if "phía trước" in text else "MEDIUM"
    }
