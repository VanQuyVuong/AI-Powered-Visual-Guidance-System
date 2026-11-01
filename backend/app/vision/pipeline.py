"""
Pipeline thị giác hợp nhất (Vision Processing Pipeline):
Kết hợp YOLOv8/11 + ByteTrack + Depth Anything V2 + SegFormer + Tactile Paving + Decision Engine
để tạo dữ liệu hướng dẫn thời gian thực cho người khiếm thị.
"""

import threading
import cv2
import numpy as np

from app.schemas.vision import Detection, BoundingBox
from app.vision.yolo_service import run_tracking
from app.vision.decision_engine import is_in_roi, determine_position, generate_guidance, get_vietnamese_name
from app.vision.tactile_paving_detector import detect_tactile_paving
from app.vision.depth_service import depth_service
from app.vision.distance_estimator import estimate_distance_m

# Danh mục các vật cản & đồ vật thực tế trong nhà và ngoài đường
OBSTACLE_CLASSES = {
    "person", "chair", "couch", "table", "dining table", "bed", "bench",
    "backpack", "handbag", "suitcase", "cell phone", "bottle", "cup",
    "wine glass", "bowl", "laptop", "tv", "keyboard", "mouse", "book",
    "potted plant", "refrigerator", "sink", "clock", "vase",
    "dog", "cat", "car", "motorcycle", "bicycle", "bus", "truck",
    "door", "stairs", "stop sign", "traffic light", "fire hydrant",
    "wall", "column", "pillar", "cabinet", "obstacle", "pothole"
}

# Các lớp đồ vật lớn
LARGE_FALSE_POSITIVE_CLASSES = {"bed", "couch"}

# YOLO tracker giữ trạng thái chung -> chỉ cho 1 luồng suy luận chạy tại một thời điểm
_inference_lock = threading.Lock()


def analyze_frame_sync(image_bytes: bytes) -> dict:
    """
    Toàn bộ phần xử lý nặng (YOLO + Depth Anything V2 + SegFormer + Pothole + Tactile Paving).
    Hàm đồng bộ chạy trong threadpool (asyncio.to_thread).
    """
    if not image_bytes:
        raise ValueError("Không thể đọc được ảnh (INVALID_IMAGE)")

    # 1. Giải mã ảnh và kích hoạt tính Depth + SegFormer
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR) if nparr.size > 0 else None
    if frame_bgr is None:
        raise ValueError("Không thể đọc được ảnh (INVALID_IMAGE)")

    depth_service.trigger_async_compute(frame_bgr)

    with _inference_lock:
        raw_detections, img_width, img_height = run_tracking(image_bytes)

    is_on_tactile = detect_tactile_paving(image_bytes)
    dmap = depth_service.get_latest_depth_map()

    all_objects = []
    roi_detections = []
    has_person_in_roi = False
    has_obstacle_in_roi = False

    for det in raw_detections:
        cname = det["class_name"].lower()
        conf = det["confidence"]
        box = det["bounding_box"]

        if cname not in OBSTACLE_CLASSES:
            continue
        if cname in (
            "cell phone", "bottle", "cup", "wine glass", "bowl", "book",
            "backpack", "handbag", "suitcase", "chair", "table", "dining table", "laptop"
        ):
            min_conf = 0.25
        elif cname == "person":
            min_conf = 0.35
        elif cname in LARGE_FALSE_POSITIVE_CLASSES:
            min_conf = 0.45
        else:
            min_conf = 0.30
        if conf < min_conf:
            continue

        in_corridor = is_in_roi(box["x1"], box["y1"], box["x2"], box["y2"], img_width, img_height)
        pos = determine_position(box["x1"], box["x2"], img_width)
        is_person = (cname == "person")

        # 2. Ưu tiên lấy khoảng cách mét thật từ Depth Anything V2 nếu có
        metric_dist = depth_service.get_box_median_depth(box["x1"], box["y1"], box["x2"], box["y2"], dmap)
        if metric_dist is not None and 0.2 <= metric_dist <= 15.0:
            dist_m = metric_dist
        else:
            dist_m = estimate_distance_m(cname, box, img_width, img_height)

        box_h_ratio = (box["y2"] - box["y1"]) / float(max(1, img_height))
        # 3. Tiêu chí nguy hiểm: Trong hành lang, ở trung tâm, chiếm kích thước đáng kể hoặc trong phạm vi <= 3.5 mét
        in_path = in_corridor or (pos == "center") or (box_h_ratio >= 0.20) or (dist_m <= 2.8)
        is_danger = in_path and (dist_m <= 3.5)

        # Chuẩn hóa confidence >= 0.40 cho các đồ vật nhỏ đã qua lọc
        effective_conf = max(conf, 0.42)

        det_item = Detection(
            class_name=det["class_name"],
            class_name_vn=get_vietnamese_name(det["class_name"]),
            confidence=effective_conf,
            position=pos,
            distance_m=dist_m,
            in_roi=in_corridor or in_path,
            is_danger=is_danger,
            bounding_box=BoundingBox(**box)
        )

        if is_danger:
            if is_person:
                has_person_in_roi = True
            else:
                has_obstacle_in_roi = True
            roi_detections.append(det_item)

        all_objects.append({
            "class_name": det["class_name"],
            "class_name_vn": get_vietnamese_name(det["class_name"]),
            "confidence": conf,
            "position": pos,
            "bounding_box": box,
            "in_roi": in_corridor or in_path,
            "distance_m": dist_m,
            "is_danger": is_danger
        })

    # 4. PHÁT HIỆN BỨC TƯỜNG, CẦU THANG, Ổ GÀ & VẬT CẢN LẠ QUA DEPTH ANYTHING V2 + SEGFORMER
    if frame_bgr is not None and dmap is not None:
        known_boxes = [d["bounding_box"] for d in raw_detections]
        unknown_obstacles = depth_service.detect_unknown_obstacles(frame_bgr, known_boxes, dmap)
        for unk in unknown_obstacles:
            all_objects.append(unk)
            if unk.get("is_danger", True):
                has_obstacle_in_roi = True
                unk_det = Detection(
                    class_name=unk["class_name"],
                    class_name_vn=unk.get("class_name_vn"),
                    confidence=max(float(unk["confidence"]), 0.45),
                    position=unk["position"],
                    distance_m=unk.get("distance_m"),
                    in_roi=True,
                    is_danger=True,
                    bounding_box=BoundingBox(**unk["bounding_box"])
                )
                roi_detections.append(unk_det)

    # 5. Nếu có vật thể mà roi_detections đang trống, đưa vật gần nhất vào để cảnh báo
    if not roi_detections and all_objects:
        nearest = sorted(all_objects, key=lambda o: o.get("distance_m") or 9.9)[:2]
        for obj in nearest:
            obj["is_danger"] = True
            roi_detections.append(
                Detection(
                    class_name=obj["class_name"],
                    class_name_vn=obj.get("class_name_vn"),
                    confidence=max(float(obj["confidence"]), 0.42),
                    position=obj["position"],
                    distance_m=obj.get("distance_m"),
                    in_roi=True,
                    is_danger=True,
                    bounding_box=BoundingBox(**obj["bounding_box"])
                )
            )

    guidance_data = generate_guidance(roi_detections) or {"text": "", "priority": "LOW"}

    if is_on_tactile:
        if guidance_data["text"]:
            guidance_data["text"] = "Đang ở trên vạch dẫn đường. " + guidance_data["text"]
        else:
            guidance_data["text"] = "Đang đi đúng vạch dẫn đường."
            guidance_data["priority"] = "LOW"
    elif not guidance_data["text"]:
        guidance_data["text"] = "Phía trước an toàn."

    return {
        "success": True,
        "objects": all_objects,
        "has_person_in_roi": has_person_in_roi,
        "has_obstacle_in_roi": has_obstacle_in_roi,
        "guidance": guidance_data,
        "img_size": {"width": img_width, "height": img_height}
    }
