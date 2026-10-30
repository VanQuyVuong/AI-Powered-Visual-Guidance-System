"""
Decision Engine – AI Visual Guidance System
Phiên bản tái thiết kế: 2 chế độ - Hướng dẫn đường đi (mặc định) & Đàm thoại (khi hỏi)
"""
import time

# ─── Priority Level Constants ────────────────────────────────────────────
CRITICAL = "CRITICAL"
HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"
NONE = "NONE"

# ─── System Mode ────────────────────────────────────────────────────────
# NAVIGATION: Chỉ hướng dẫn đường đi (đi thẳng, rẽ trái/phải)
# CONVERSATION: Trả lời chi tiết về vật cản khi người dùng hỏi
SYSTEM_MODE = "NAVIGATION"  # Default mode

def set_mode(mode: str):
    """Chuyển chế độ hệ thống"""
    global SYSTEM_MODE
    if mode in ["NAVIGATION", "CONVERSATION"]:
        changed = SYSTEM_MODE != mode
        SYSTEM_MODE = mode
        # Đổi mode thì cho phép nói câu mới ngay (không bị cooldown NAV nuốt)
        if changed:
            reset_alert_cooldown()

def get_mode() -> str:
    """Lấy chế độ hiện tại"""
    return SYSTEM_MODE

# ─── Cooldown tracker ──────────────────────────────────────────────────
# Lưu thời điểm cuối cùng từng object (theo id hoặc class+pos) được cảnh báo
_last_alerted: dict = {}
COOLDOWN_SECONDS = 5.0   # Tăng lên 5 giây để tránh spam audio

# ─── User Position Tracking ─────────────────────────────────────────────
# Track vị trí người dùng theo thời gian để hướng dẫn từ từ
_user_position_history = []  # List of {x, y, timestamp}
MAX_HISTORY = 10  # Giữ 10 vị trí gần nhất

def update_user_position(x: float, y: float):
    """Cập nhật vị trí người dùng (tính từ camera hoặc tracking)"""
    _user_position_history.append({
        'x': x,
        'y': y,
        'timestamp': time.time()
    })
    if len(_user_position_history) > MAX_HISTORY:
        _user_position_history.pop(0)

def get_user_movement_direction() -> str:
    """Xác định hướng di chuyển người dùng dựa trên lịch sử vị trí"""
    if len(_user_position_history) < 2:
        return "unknown"
    
    recent = _user_position_history[-1]
    older = _user_position_history[0]
    
    dx = recent['x'] - older['x']
    dy = recent['y'] - older['y']
    
    # Xác định hướng dựa trên displacement lớn nhất
    if abs(dx) > abs(dy):
        if dx > 0.1:
            return "right"
        elif dx < -0.1:
            return "left"
    else:
        if dy > 0.1:
            return "forward"
        elif dy < -0.1:
            return "backward"
    
    return "stationary"


def _object_key(class_name: str, position: str, obj_id=None) -> str:
    """Tạo key duy nhất cho từng vật thể để kiểm soát cooldown."""
    if obj_id is not None:
        return f"id_{obj_id}"
    return f"{class_name}_{position}"


def should_alert(class_name: str, position: str, obj_id=None) -> bool:
    """Trả về True nếu đây là lần đầu hoặc đã quá cooldown."""
    key = _object_key(class_name, position, obj_id)
    now = time.time()
    last = _last_alerted.get(key, 0)
    if now - last >= COOLDOWN_SECONDS:
        _last_alerted[key] = now
        return True
    return False


def reset_alert_cooldown():
    """Xóa cooldown cảnh báo — dùng khi đổi NAV ↔ CHAT."""
    _last_alerted.clear()


def cleanup_stale_keys():
    """Dọn dẹp các key cũ (quá 30 giây) để tránh memory leak."""
    now = time.time()
    stale = [k for k, t in _last_alerted.items() if now - t > 30]
    for k in stale:
        del _last_alerted[k]


# ─── Position ──────────────────────────────────────────────────────────
def determine_position(x1: int, x2: int, image_width: int = 640) -> str:
    """Phân vùng trái / giữa / phải theo tâm bounding box."""
    center_x = (x1 + x2) / 2
    third = image_width / 3
    if center_x < third:
        return "left"
    elif center_x > 2 * third:
        return "right"
    else:
        return "center"


# ─── ROI ───────────────────────────────────────────────────────────────
def is_in_roi(x1: int, y1: int, x2: int, y2: int, img_width: int, img_height: int) -> bool:
    """
    Vùng Quan Tâm hình thang: chỉ cảnh báo vật thể nằm trong
    'hành lang đi bộ' (60% phía dưới, bỏ qua 15% hai rìa ngang).
    """
    obj_bottom_x = (x1 + x2) / 2
    obj_bottom_y = y2

    # Lọc theo chiều cao: bỏ qua vật quá cao (phía chân trời / trời)
    if obj_bottom_y < img_height * 0.22:
        return False

    # Lọc theo chiều ngang: bỏ qua nếu quá sát rìa
    left_margin  = img_width * 0.08
    right_margin = img_width * 0.92
    if obj_bottom_x < left_margin or obj_bottom_x > right_margin:
        return False

    return True


# ─── Distance estimate (Perspective-based, fallback) ─────────────────
def estimate_distance(x1: int, y1: int, x2: int, y2: int,
                      img_width: int, img_height: int) -> dict:
    """
    Ước lượng độ sâu theo nguyên lý PHỐI CẢNH (fallback khi không có depth map):
    - Điểm chạm đất = đáy bounding box (y2).
    - Vật ở đáy khung hình → GẦN; gần chân trời → XA.
    """
    ground_y    = y2 / img_height
    obj_h_ratio = (y2 - y1) / img_height
    score = ground_y * 0.75 + obj_h_ratio * 0.25

    if score > 0.80:
        return {"label": "rất gần",    "level": "CRITICAL", "meters": "<1m",   "depth_zone": 1, "source": "perspective"}
    elif score > 0.65:
        return {"label": "khá gần",    "level": "HIGH",     "meters": "~1-2m", "depth_zone": 2, "source": "perspective"}
    elif score > 0.50:
        return {"label": "trung bình", "level": "MEDIUM",   "meters": "~2-5m", "depth_zone": 3, "source": "perspective"}
    elif score > 0.38:
        return {"label": "còn xa",     "level": "LOW",      "meters": "~5-8m", "depth_zone": 4, "source": "perspective"}
    else:
        return {"label": "rất xa",     "level": "NONE",     "meters": ">8m",   "depth_zone": 5, "source": "perspective"}


def estimate_distance_from_depth(depth_value: float) -> dict:
    """
    Chuyển đổi giá trị depth map [0.0 xa .. 1.0 gần]
    từ Depth Anything V2 → nhãn khoảng cách.
    Độ chính xác cao hơn estimate_distance() phối cảnh.
    """
    if depth_value < 0:
        # -1.0 = không có depth map, fallback
        return {"label": "không rõ", "level": "NONE", "meters": "?", "depth_zone": 0, "source": "none"}

    d = depth_value  # 1.0 = rất gần, 0.0 = rất xa

    if d > 0.82:
        return {"label": "rất gần",    "level": "CRITICAL", "meters": "<1m",   "depth_zone": 1, "source": "depth_ai"}
    elif d > 0.65:
        return {"label": "khá gần",    "level": "HIGH",     "meters": "~1-2m", "depth_zone": 2, "source": "depth_ai"}
    elif d > 0.48:
        return {"label": "trung bình", "level": "MEDIUM",   "meters": "~2-5m", "depth_zone": 3, "source": "depth_ai"}
    elif d > 0.30:
        return {"label": "còn xa",     "level": "LOW",      "meters": "~5-8m", "depth_zone": 4, "source": "depth_ai"}
    else:
        return {"label": "rất xa",     "level": "NONE",     "meters": ">8m",   "depth_zone": 5, "source": "depth_ai"}


# ─── Object clustering (gom nhóm chồng lấp) ───────────────────────────
# Nhóm cha-con: nếu object A nằm gần hoàn toàn bên trong object B (IOU cao
# hoặc containment > threshold) thì giữ object B (bao quát hơn).
_PRIORITY_CLASSES = [
    # Nhóm con người & phương tiện được ưu tiên làm nhãn bao quát
    "person", "car", "truck", "bus", "motorcycle", "bicycle",
    "dog", "cat", "bird",
]


def _iou(a: dict, b: dict) -> float:
    """Tính IoU giữa 2 bounding box dict {x1,y1,x2,y2}."""
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    aA = (ax2-ax1) * (ay2-ay1)
    bA = (bx2-bx1) * (by2-by1)
    return inter / (aA + bA - inter + 1e-6)


def _containment(inner: dict, outer: dict) -> float:
    """Tỷ lệ diện tích inner nằm trong outer (0..1)."""
    ax1, ay1, ax2, ay2 = inner["x1"], inner["y1"], inner["x2"], inner["y2"]
    bx1, by1, bx2, by2 = outer["x1"], outer["y1"], outer["x2"], outer["y2"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    aA = (ax2-ax1) * (ay2-ay1)
    if aA < 1:
        return 0.0
    return inter / aA


def cluster_objects(raw_detections: list) -> list:
    """
    Gom nhóm các vật thể chồng lấp, giữ lại nhãn bao quát hơn.
    Ví dụ: tie/backpack/handbag nằm trong bbox của person → loại bỏ, chỉ giữ person.

    Args:
        raw_detections: list dict từ yolo_service (có key bounding_box)
    Returns:
        list dict đã lọc
    """
    if not raw_detections:
        return []

    # Ưu tiên: class trong _PRIORITY_CLASSES có điểm cao hơn
    def priority(det):
        cls = det.get("class_name", "").lower()
        try:
            return -_PRIORITY_CLASSES.index(cls)   # âm → sort tăng dần
        except ValueError:
            return 1  # không có trong list → ưu tiên thấp

    dets = sorted(raw_detections, key=lambda d: priority(d))

    kept = []  # index của dets được giữ lại
    suppressed = set()

    for i, det_i in enumerate(dets):
        if i in suppressed:
            continue
        box_i = det_i.get("bounding_box", {})
        for j, det_j in enumerate(dets):
            if i == j or j in suppressed:
                continue
            box_j = det_j.get("bounding_box", {})
            # Nếu j nằm hầu hết bên trong i (j là chi tiết, i bao quát)
            if _containment(box_j, box_i) > 0.60:
                suppressed.add(j)  # loại j (chi tiết)
            # Nếu IoU quá cao và i có priority cao hơn
            elif _iou(box_i, box_j) > 0.55:
                if priority(det_i) <= priority(det_j):
                    suppressed.add(j)
        kept.append(i)

    return [dets[i] for i in kept if i not in suppressed]


# ─── Vocabulary ────────────────────────────────────────────────────────
VOCAB_VI = {
    "person":        "người",
    "car":           "xe ô tô",
    "motorcycle":    "xe máy",
    "bicycle":       "xe đạp",
    "bus":           "xe buýt",
    "truck":         "xe tải",
    "dog":           "con chó",
    "cat":           "con mèo",
    "chair":         "cái ghế",
    "bottle":        "chai nước",
    "table":         "cái bàn",
    "laptop":        "máy tính xách tay",
    "phone":         "điện thoại",
    "stop sign":     "biển báo dừng",
    "traffic light": "đèn giao thông",
    "tv":            "tivi",
    "tie":           "cà vạt",
    "cup":           "cốc",
    "backpack":      "ba lô",
    "umbrella":      "cái dù",
    "handbag":       "túi xách",
    "suitcase":      "vali",
    "bed":           "giường",
    "couch":         "sofa",
    "potted plant":  "cây cảnh",
    "clock":         "đồng hồ",
    "vase":          "lọ hoa",
    "scissors":      "kéo",
    "teddy bear":    "gấu bông",
    "hair drier":    "máy sấy tóc",
    "toothbrush":    "bàn chải đánh răng",
    "mirror":        "gương",
    "sink":          "bồn rửa",
    "refrigerator":  "tủ lạnh",
    "book":          "sách",
    "mouse":         "chuột máy tính",
    "keyboard":      "bàn phím",
    "cell phone":    "điện thoại",
    "remote":        "điều khiển",
    "microwave":     "lò vi sóng",
    "oven":          "lò nướng",
    "toaster":       "máy nướng bánh",
    "sink":          "bồn rửa",
    "toilet":        "nhà vệ sinh",
}

def _vn(class_name: str) -> str:
    return VOCAB_VI.get(class_name.lower(), class_name)


# ─── Navigation Guidance Templates (Mặc định - chỉ hướng dẫn đường đi) ───
# Chỉ nói cách đi, không nhắc đến vật cản - Tự nhiên hơn
_NAV_TEMPLATES_STRAIGHT = [
    "Đi thẳng nhé.",
    "Cứ đi thẳng tiếp đi.",
    "Đường phía trước an toàn, đi thẳng.",
    "Tiếp tục đi thẳng.",
]
_NAV_TEMPLATES_LEFT = [
    "Rẽ trái một chút nhé.",
    "Sang trái đi.",
    "Bên trái an toàn hơn, rẽ trái đi.",
    "Rẽ trái 3 bước.",
]
_NAV_TEMPLATES_RIGHT = [
    "Rẽ phải một chút nhé.",
    "Sang phải đi.",
    "Bên phải an toàn hơn, rẽ phải đi.",
    "Rẽ phải 3 bước.",
]
_NAV_TEMPLATES_STOP = [
    "Dừng lại nhé.",
    "Dừng lại cẩn thận.",
    "Có vật cản, dừng lại.",
    "Dừng ngay đi.",
]

# ─── Conversation Templates (Khi người dùng hỏi - mô tả vật cản) ────────
_CONV_TEMPLATES_CENTER_CRITICAL = [
    "Cẩn thận, ngay trước mặt có {name} rất gần, {dist}.",
    "Ngay trước bạn có {name}, {dist}. Hãy chú ý.",
    "Phía trước có {name} rất gần, cách khoảng {dist}.",
]
_CONV_TEMPLATES_CENTER_HIGH = [
    "Phía trước có {name}, khoảng {dist}.",
    "Thấy {name} ở trước, {dist}.",
    "Có {name} phía trước, khoảng {dist}.",
]
_CONV_TEMPLATES_SIDE_CRITICAL = [
    "{side} có {name} rất gần, {dist}. Cẩn thận nhé.",
    "{side} phát hiện {name}, {dist}.",
    "Bên {side.lower()} có {name} rất gần, {dist}.",
]
_CONV_TEMPLATES_SIDE_HIGH = [
    "{side} có {name}, khoảng {dist}.",
    "Bên {side.lower()} có {name}, {dist}.",
]
_CONV_TEMPLATES_SAFE = [
    "Đường thông thoáng, không có vật cản.",
    "An toàn, phía trước và hai bên đều trống.",
    "Không có vật cản, đi thoải mái nhé.",
]
# Template cho vật cản chưa xác định (chỉ depth phát hiện)
_CONV_TEMPLATES_UNKNOWN_CRITICAL = [
    "Cẩn thận, phía trước có vật cản rất gần nhưng chưa xác định được.",
    "Ngay trước mặt có vật cản, {dist}. Bạn có thể nói rõ là gì không?",
    "Phía trước có vật cản rất gần, {dist}. Cần bạn mô tả thêm.",
]
_CONV_TEMPLATES_UNKNOWN_HIGH = [
    "Phía trước có vật cản khoảng {dist}, chưa xác định được.",
    "Có vật cản phía trước, {dist}. Bạn thấy là gì?",
    "Phía trước có vật cản, {dist}.",
]

# ─── Q&A System ─────────────────────────────────────────────────────────
# Hệ thống trả lời câu hỏi tự động dựa trên dữ liệu phát hiện
def answer_question(question: str, detections: list, img_width: int = 640, img_height: int = 480) -> dict:
    """
    Trả lời câu hỏi của người dùng dựa trên dữ liệu phát hiện hiện tại.
    Returns: {"text": "...", "priority": "..."}
    """
    if not question or not question.strip():
        return {"text": "Bạn muốn hỏi gì?", "priority": "LOW"}

    question = question.lower().strip()
    valid = [d for d in detections if d.get("confidence", 0) >= 0.20]

    # Phân loại câu hỏi
    if any(word in question for word in ["trước", "phía trước", "trước mặt"]):
        return _answer_about_front(valid, question, img_width, img_height)
    elif any(word in question for word in ["trái", "bên trái"]):
        return _answer_about_side(valid, "left", question)
    elif any(word in question for word in ["phải", "bên phải"]):
        return _answer_about_side(valid, "right", question)
    elif any(word in question for word in ["cái gì", "là gì", "vật cản", "đó là"]):
        return _answer_what_is(valid, question)
    elif any(word in question for word in ["gần", "xa", "khoảng cách"]):
        return _answer_distance(valid, question)
    elif any(word in question for word in ["an toàn", "đi được", "thông thoáng"]):
        return _answer_safety(valid, question)
    else:
        return _answer_general(valid, question)

def _answer_about_front(detections: list, question: str, img_width: int, img_height: int) -> dict:
    """Trả lời về phía trước"""
    front_objs = [d for d in detections if d.get("position") == "center"]
    
    if not front_objs:
        return {"text": "Phía trước không có vật cản nào, an toàn để đi.", "priority": "LOW"}
    
    # Tìm vật gần nhất
    closest = min(front_objs, key=lambda d: d.get("distance", {}).get("depth_zone", 5))
    name = _vn(closest.get("class_name"))
    dist = closest.get("distance", {}).get("label", "không rõ")
    
    if len(front_objs) == 1:
        return {"text": f"Phía trước có {name}, cách khoảng {dist}.", "priority": "MEDIUM"}
    else:
        return {"text": f"Phía trước có {len(front_objs)} vật thể, gần nhất là {name} {dist}.", "priority": "MEDIUM"}

def _answer_about_side(detections: list, side: str, question: str) -> dict:
    """Trả lời về bên trái/phải"""
    side_objs = [d for d in detections if d.get("position") == side]
    side_name = "bên trái" if side == "left" else "bên phải"
    
    if not side_objs:
        return {"text": f"{side_name.capitalize()} không có vật cản.", "priority": "LOW"}
    
    closest = min(side_objs, key=lambda d: d.get("distance", {}).get("depth_zone", 5))
    name = _vn(closest.get("class_name"))
    dist = closest.get("distance", {}).get("label", "không rõ")
    
    return {"text": f"{side_name.capitalize()} có {name}, cách khoảng {dist}.", "priority": "MEDIUM"}

def _answer_what_is(detections: list, question: str) -> dict:
    """Trả lời vật thể là gì"""
    if not detections:
        return {"text": "Không phát hiện vật thể nào gần đây.", "priority": "LOW"}
    
    # Tìm vật nguy hiểm nhất
    PRIO = {CRITICAL:0, HIGH:1, MEDIUM:2, LOW:3, NONE:4}
    most_dangerous = min(detections, key=lambda d: PRIO.get(d.get("distance", {}).get("level", "NONE"), 4))
    
    name = _vn(most_dangerous.get("class_name"))
    pos = most_dangerous.get("position")
    dist = most_dangerous.get("distance", {}).get("label", "không rõ")
    
    pos_name = "trước mặt" if pos == "center" else f"bên {'trái' if pos == 'left' else 'phải'}"
    
    return {"text": f"Vật thể nguy hiểm nhất là {name}, nằm {pos_name}, cách khoảng {dist}.", "priority": "HIGH"}

def _answer_distance(detections: list, question: str) -> dict:
    """Trả lời về khoảng cách"""
    if not detections:
        return {"text": "Không có vật thể nào để đo khoảng cách.", "priority": "LOW"}
    
    closest = min(detections, key=lambda d: d.get("distance", {}).get("depth_zone", 5))
    name = _vn(closest.get("class_name"))
    dist = closest.get("distance", {}).get("label", "không rõ")
    meters = closest.get("distance", {}).get("meters", "?")
    
    return {"text": f"Vật thể gần nhất là {name}, cách khoảng {meters}.", "priority": "MEDIUM"}

def _answer_safety(detections: list, question: str) -> dict:
    """Trả lời về độ an toàn"""
    dangerous = [d for d in detections if d.get("in_roi") and d.get("distance", {}).get("level") in ["CRITICAL", "HIGH"]]
    
    if not dangerous:
        return {"text": "Đường đi an toàn, không có vật cản nguy hiểm.", "priority": "LOW"}
    
    count = len(dangerous)
    return {"text": f"Cẩn thận, có {count} vật cản nguy hiểm gần bạn.", "priority": "HIGH"}

def _answer_general(detections: list, question: str) -> dict:
    """Trả lời câu hỏi chung"""
    if not detections:
        return {"text": "Không phát hiện vật thể nào xung quanh.", "priority": "LOW"}
    
    count = len(detections)
    return {"text": f"Hiện tại phát hiện {count} vật thể xung quanh bạn.", "priority": "LOW"}

import random
_template_counters: dict = {}

def _pick_template(templates: list, key: str) -> str:
    """Lần lượt chọn template để tránh lặp lại cùng một câu."""
    idx = _template_counters.get(key, 0)
    result = templates[idx % len(templates)]
    _template_counters[key] = (idx + 1) % len(templates)
    return result


def generate_natural_guidance(detections: list, img_width: int = 640, img_height: int = 480, depth_map=None) -> dict:
    """
    Tạo câu hướng dẫn dựa trên chế độ hệ thống:
    - NAVIGATION: Chỉ hướng dẫn đường đi (đi thẳng, rẽ trái/phải, dừng)
    - CONVERSATION: Mô tả chi tiết vật cản khi người dùng hỏi
    - depth_map: numpy array depth map để phát hiện vật cẩn khi YOLO không nhận diện
    """
    cleanup_stale_keys()

    # Lọc detection đủ confidence
    valid = [d for d in detections if d.get("confidence", 0) >= 0.20]

    # CHECK: Nếu YOLO không phát hiện vật cản nhưng depth phát hiện vùng gần
    if not valid and depth_map is not None:
        # Check vùng center của depth map xem có vật cản gần không
        # depth_value: 0.0 (xa) -> 1.0 (gần)
        h, w = depth_map.shape[:2] if hasattr(depth_map, 'shape') else (img_height, img_width)
        center_region = depth_map[int(h*0.5):int(h*0.8), int(w*0.3):int(w*0.7)]
        if center_region.size > 0:
            avg_depth = center_region.mean()
            # Nếu depth > 0.7 nghĩa là có vật cản rất gần
            if avg_depth > 0.7:
                if SYSTEM_MODE == "NAVIGATION":
                    return {"text": "Phía trước có vật cản, hãy dừng lại.", "priority": "HIGH"}
                else:
                    # Conversation mode: hỏi user
                    dist = estimate_distance_from_depth(avg_depth)
                    dist_label = dist.get("label", "khá gần")
                    dist_level = dist.get("level", "HIGH")
                    if dist_level in ("CRITICAL", "HIGH"):
                        tpl = _pick_template(_CONV_TEMPLATES_UNKNOWN_CRITICAL, "conv_unknown_crit")
                        return {"text": tpl.format(dist=dist_label), "priority": dist_level}
                    else:
                        tpl = _pick_template(_CONV_TEMPLATES_UNKNOWN_HIGH, "conv_unknown_high")
                        return {"text": tpl.format(dist=dist_label), "priority": dist_level}

    if not valid:
        if SYSTEM_MODE == "NAVIGATION":
            # Tùy theo hướng di chuyển để điều chỉnh câu
            movement = get_user_movement_direction()
            if movement == "right":
                return {"text": "Bạn đang đi sang phải, tiếp tục đi nhé.", "priority": "LOW"}
            elif movement == "left":
                return {"text": "Bạn đang đi sang trái, tiếp tục đi nhé.", "priority": "LOW"}
            elif movement == "forward":
                return {"text": "Bạn đang đi thẳng, cứ đi tiếp nhé.", "priority": "LOW"}
            else:
                return {"text": "Đi thẳng nhé.", "priority": "LOW"}
        else:
            return {"text": "Không phát hiện vật cản.", "priority": "LOW"}

    # Phân loại theo vùng và mức độ nguy hiểm
    center_objs = [d for d in valid if d.get("position") == "center"]
    left_objs = [d for d in valid if d.get("position") == "left"]
    right_objs = [d for d in valid if d.get("position") == "right"]

    PRIO = {CRITICAL:0, HIGH:1, MEDIUM:2, LOW:3, NONE:4}

    # Tính toán danger score cho mỗi vùng
    # CRITICAL=4, HIGH=3, MEDIUM=2, LOW=1, NONE=0
    def get_danger_score(objs):
        return sum((4 - PRIO.get((d.get("distance") or {}).get('level') or 'NONE', 4)) for d in objs)

    center_score = get_danger_score(center_objs)
    left_score = get_danger_score(left_objs)
    right_score = get_danger_score(right_objs)

    # Tìm vật thể nguy hiểm nhất (cho conversation mode)
    most_dangerous = max(valid, key=lambda d: (4 - PRIO.get((d.get("distance") or {}).get('level') or 'NONE', 4)))

    if SYSTEM_MODE == "NAVIGATION":
        # CHẾ ĐỘ HƯỚNG DẪN: Chỉ nói cách đi

        # CASE 1: Cả 3 hướng đều nguy hiểm → DỪNG LẠI
        if center_score >= 2 and left_score >= 2 and right_score >= 2:
            tpl = _pick_template(_NAV_TEMPLATES_STOP, "nav_stop")
            return {"text": tpl, "priority": "CRITICAL"}

        # CASE 2: Center bị chặn nặng → rẽ sang hướng an toàn hơn
        if center_score >= 2:
            if left_score <= right_score:
                tpl = _pick_template(_NAV_TEMPLATES_LEFT, "nav_left")
                return {"text": tpl, "priority": "HIGH"}
            else:
                tpl = _pick_template(_NAV_TEMPLATES_RIGHT, "nav_right")
                return {"text": tpl, "priority": "HIGH"}

        # CASE 3: Center có vật cản (score >= 1, tức ít nhất 1 vật LOW+) → rẽ nhẹ
        elif center_score >= 1:
            if left_score <= right_score:
                tpl = _pick_template(_NAV_TEMPLATES_LEFT, "nav_left")
                return {"text": tpl, "priority": "MEDIUM"}
            else:
                tpl = _pick_template(_NAV_TEMPLATES_RIGHT, "nav_right")
                return {"text": tpl, "priority": "MEDIUM"}

        # CASE 4: Center an toàn → đi thẳng
        else:
            tpl = _pick_template(_NAV_TEMPLATES_STRAIGHT, "nav_straight")
            return {"text": tpl, "priority": "LOW"}

    else:  # CONVERSATION MODE
        # CHẾ ĐỘ ĐÀM THOẠI: luôn trả text để UI đổi ngay.
        # TTS tự lọc câu trùng (queue_guidance_tts), không nuốt guidance bằng None.
        obj_id = most_dangerous.get("id")
        should_alert(most_dangerous.get("class_name"), most_dangerous.get("position"), obj_id)

        name = _vn(most_dangerous.get("class_name"))

        # Dùng distance info có sẵn trong detection (đã tính ở _build_objects_payload)
        dist_info = most_dangerous.get("distance")
        if dist_info and dist_info.get("label"):
            dist_label = dist_info["label"]
            dist_level = dist_info.get("level", "MEDIUM")
        else:
            # Fallback: tính từ bounding box
            box = most_dangerous.get("bounding_box", {})
            dist = estimate_distance(box.get("x1", 0), box.get("y1", 0), box.get("x2", 0), box.get("y2", 0), img_width, img_height)
            dist_label = dist["label"]
            dist_level = dist["level"]

        # Chọn template dựa trên vị trí và mức độ
        if most_dangerous.get("position") == "center":
            if dist_level in ("CRITICAL", "HIGH"):
                tpl = _pick_template(_CONV_TEMPLATES_CENTER_CRITICAL, "conv_center_crit")
                return {"text": tpl.format(name=name, dist=dist_label), "priority": dist_level}
            else:
                tpl = _pick_template(_CONV_TEMPLATES_CENTER_HIGH, "conv_center_high")
                return {"text": tpl.format(name=name, dist=dist_label), "priority": dist_level}
        else:
            side = "Bên trái" if most_dangerous.get("position") == "left" else "Bên phải"
            if dist_level in ("CRITICAL", "HIGH"):
                tpl = _pick_template(_CONV_TEMPLATES_SIDE_CRITICAL, "conv_side_crit")
                return {"text": tpl.format(side=side, name=name, dist=dist_label), "priority": dist_level}
            else:
                tpl = _pick_template(_CONV_TEMPLATES_SIDE_HIGH, "conv_side_high")
                return {"text": tpl.format(side=side, name=name, dist=dist_label), "priority": dist_level}


# Backward-compatible alias (main.py dùng generate_guidance)
def generate_guidance(detections: list, img_width: int = 640, img_height: int = 480, depth_map=None) -> dict:
    return generate_natural_guidance(detections, img_width, img_height, depth_map)


def generate_navigation_direction(detections: list, img_width: int = 640, img_height: int = 480) -> dict:
    """
    Tạo hướng dẫn đường đi cho Panel 3 (không phụ thuộc mode, luôn có)
    Trả về: {"direction": "straight"|"left"|"right"|"stop", "confidence": 0-1,
             "scores": {"center": int, "left": int, "right": int}}
    detections: list of dict từ objects_payload (có key: position, distance, confidence)
    """
    if not detections:
        return {"direction": "straight", "confidence": 0.9, "scores": {"center": 0, "left": 0, "right": 0}}

    center_objs = [d for d in detections if d.get("position") == "center"]
    left_objs = [d for d in detections if d.get("position") == "left"]
    right_objs = [d for d in detections if d.get("position") == "right"]

    PRIO = {CRITICAL:0, HIGH:1, MEDIUM:2, LOW:3, NONE:4}

    def get_danger_score(objs):
        return sum((4 - PRIO.get((d.get("distance") or {}).get('level') or 'NONE', 4)) for d in objs)

    center_score = get_danger_score(center_objs)
    left_score = get_danger_score(left_objs)
    right_score = get_danger_score(right_objs)
    scores = {"center": center_score, "left": left_score, "right": right_score}

    # Cả 3 hướng đều nguy hiểm → STOP
    if center_score >= 2 and left_score >= 2 and right_score >= 2:
        return {"direction": "stop", "confidence": 0.95, "scores": scores}

    if center_score >= 2:
        # Center bị chặn nặng
        if left_score <= right_score:
            return {"direction": "left", "confidence": min(0.9, 0.5 + center_score * 0.1), "scores": scores}
        else:
            return {"direction": "right", "confidence": min(0.9, 0.5 + center_score * 0.1), "scores": scores}
    elif center_score >= 1:
        # Center có vật cản nhẹ
        if left_score <= right_score:
            return {"direction": "left", "confidence": 0.6, "scores": scores}
        else:
            return {"direction": "right", "confidence": 0.6, "scores": scores}
    else:
        # Center an toàn
        return {"direction": "straight", "confidence": 0.9, "scores": scores}
