"""
Bộ kiểm thử cho các tối ưu tốc độ thời gian thực, điều phối đa mô hình và giảm trễ mạng.
"""

import time
import cv2
import numpy as np


def test_model_scheduler_mutual_exclusion_and_threads():
    from app.vision.model_scheduler import AuxModelCoordinator, configure_cpu_threads

    threads = configure_cpu_threads()
    assert threads >= 0

    coord = AuxModelCoordinator()
    assert coord.current_owner() is None
    assert coord.try_acquire("depth") is True
    assert coord.current_owner() == "depth"
    # Khi depth đang chiếm quyền, edgesam không được chạy đè lên cùng lúc
    assert coord.try_acquire("edgesam") is False
    # Nhả sai chủ sở hữu không làm mất khóa
    coord.release("edgesam")
    assert coord.current_owner() == "depth"
    coord.release("depth")
    assert coord.current_owner() is None
    assert coord.try_acquire("edgesam") is True
    coord.release("edgesam")


def test_fusion_engine_returns_fast_latency_and_aabb_dedup():
    from app.vision.fusion_engine import MultiModelFusionEngine

    engine = MultiModelFusionEngine()
    yolo_objs = [
        {
            "class_name": "chair",
            "class_name_vn": "ghế",
            "confidence": 0.88,
            "position": "center",
            "distance_m": 1.4,
            "in_roi": True,
            "is_danger": True,
            "bounding_box": {"x1": 210, "y1": 160, "x2": 390, "y2": 430},
        },
        {
            "class_name": "person",
            "class_name_vn": "người",
            "confidence": 0.92,
            "position": "right",
            "distance_m": 2.0,
            "in_roi": False,
            "is_danger": False,
            "bounding_box": {"x1": 490, "y1": 80, "x2": 620, "y2": 460},
        },
    ]
    dup_wall = [
        {
            "class_name": "wall",
            "class_name_vn": "bức tường",
            "confidence": 0.78,
            "position": "center",
            "distance_m": 1.4,
            "in_roi": True,
            "is_danger": False,
            "bounding_box": {"x1": 200, "y1": 150, "x2": 400, "y2": 440},
        }
    ]
    res = engine.fuse_detections(yolo_objs, dup_wall, [], None, 640, 480, now=100.0)
    summary = res["fusion_summary"]

    assert "fusion_latency_ms" in summary
    assert 0.0 <= summary["fusion_latency_ms"] < 20.0
    assert summary["dedup_removed_count"] == 1
    assert len(res["fused_objects"]) == 2


def test_sam_service_cached_masks_and_router_fast_path():
    from app.vision.sam_service import sam_service
    from app.vision.sam_router import _segment_sync

    snap_masks = list(sam_service._latest_masks)
    snap_time = sam_service._latest_masks_time
    snap_size = sam_service._latest_img_size
    try:
        sam_service._latest_masks = [
            {
                "area_px": 18000,
                "area_ratio": 0.12,
                "bounding_box": {"x1": 120, "y1": 100, "x2": 280, "y2": 260},
                "polygon": [[120, 100], [280, 100], [280, 260], [120, 260]],
            }
        ]
        sam_service._latest_masks_time = time.time()
        sam_service._latest_img_size = (640, 480)

        cached = sam_service.get_cached_masks(max_age_s=2.5)
        assert cached is not None and len(cached) == 1

        frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", frame)
        res = _segment_sync(buf.tobytes(), use_cached_if_fresh=True)
        assert res["success"] is True
        assert res["cached"] is True
        assert res["count"] == 1
        assert res["overlay_b64"]
    finally:
        sam_service._latest_masks = snap_masks
        sam_service._latest_masks_time = snap_time
        sam_service._latest_img_size = snap_size


def test_schema_light_client_payload_strips_heavy_images():
    from app.schemas.vision import build_light_client_payload

    full_payload = {
        "success": True,
        "mode": "NAVIGATION",
        "processing_ms": 82.4,
        "objects": [{"class_name": "chair", "position": "center"}],
        "has_person_in_roi": False,
        "has_obstacle_in_roi": True,
        "guidance": {"text": "Phía trước có ghế.", "priority": "HIGH"},
        "navigation": {
            "direction": "right",
            "confidence": 0.9,
            "text": "Rẽ phải",
            "fusion_summary": {"fusion_latency_ms": 0.2},
        },
        "frame_b64": "HEAVY_FRAME_BASE64",
        "depth_b64": "HEAVY_DEPTH_BASE64",
        "depth": {"colormap_b64": "HEAVY_DEPTH_BASE64"},
        "img_size": {"width": 640, "height": 480},
    }
    light = build_light_client_payload(full_payload)
    assert "frame_b64" not in light
    assert "depth_b64" not in light
    assert "depth" not in light
    assert "fusion_summary" not in light["navigation"]
    assert light["navigation"]["direction"] == "right"
    assert light["guidance"]["text"] == "Phía trước có ghế."
    assert light["img_size"] == {"width": 640, "height": 480}


def test_guided_scan_rapid_urgent_reroute_and_fast_post_turn_recovery():
    from app.vision.guided_scan import reset_voice_state, update_spatial_scan_and_voice

    reset_voice_state()
    w = 640

    # Bước 1: Có ghế ở giữa -> Trái thoáng -> Báo rẽ trái
    chair = [
        {
            "class_name": "chair",
            "class_name_vn": "ghế",
            "position": "center",
            "distance_m": 1.3,
            "in_roi": True,
            "bounding_box": {"x1": 220, "y1": 160, "x2": 420, "y2": 440},
        }
    ]
    lanes_left_open = {
        "has_depth": True,
        "lanes": {
            "left": {"median_m": 3.2, "is_blocked": False},
            "center": {"median_m": 1.3, "is_blocked": True},
            "right": {"median_m": 1.0, "is_blocked": True},
        },
    }
    sp1, _, ok1 = update_spatial_scan_and_voice("Phía trước có ghế.", "HIGH", chair, lanes_left_open, w, 10.0)
    assert ok1 is True and "rẽ sang trái" in sp1

    # Bước 2 (chỉ 0.3s sau): Vật cản sát bên trái 1.0m, bên phải mở -> Đổi hướng rẽ phải ngay lập tức
    urgent_person = [
        {
            "class_name": "chair",
            "class_name_vn": "ghế",
            "position": "center",
            "distance_m": 1.1,
            "in_roi": True,
            "bounding_box": {"x1": 220, "y1": 160, "x2": 420, "y2": 440},
        },
        {
            "class_name": "person",
            "class_name_vn": "người",
            "position": "left",
            "distance_m": 1.0,
            "in_roi": False,
            "bounding_box": {"x1": 10, "y1": 100, "x2": 180, "y2": 460},
        },
    ]
    lanes_right_open = {
        "has_depth": True,
        "lanes": {
            "left": {"median_m": 1.0, "is_blocked": True},
            "center": {"median_m": 1.1, "is_blocked": True},
            "right": {"median_m": 3.5, "is_blocked": False},
        },
    }
    sp2, _, ok2 = update_spatial_scan_and_voice("Phía trước có ghế.", "HIGH", urgent_person, lanes_right_open, w, 10.3)
    assert ok2 is True and "rẽ sang phải" in sp2

    # Bước 3: Rẽ phải xong -> Báo đi thẳng an toàn sau 0.6s (2 frames)
    lanes_clear = {
        "has_depth": True,
        "lanes": {
            "left": {"median_m": 3.4, "is_blocked": False},
            "center": {"median_m": 3.6, "is_blocked": False},
            "right": {"median_m": 3.4, "is_blocked": False},
        },
    }
    update_spatial_scan_and_voice("", "LOW", [], lanes_clear, w, 10.6)
    sp3, prio3, ok3 = update_spatial_scan_and_voice("", "LOW", [], lanes_clear, w, 10.95)
    assert ok3 is True and sp3 == "Phía trước an toàn, đi thẳng." and prio3 == "LOW"


def test_distance_calibration_portrait_chair_4_to_5_tiles():
    """
    Ghế ngoài đời cách 4-5 viên gạch (~2.6m - 3.0m) trên khung hình dọc điện thoại 480x640:
    - Pinhole KHÔNG được phép bóp trần xuống 2.0m khi box_h > 25%.
    - Nếu Depth Anything V2 bị kéo tụt xuống ~1.8m (do sàn trước mặt hoặc sai scale),
      `resolve_calibrated_distance` phải giữ được >= 2.4m để không báo vật cản giả ở xa.
    """
    from app.vision.distance_estimator import estimate_distance_m, resolve_calibrated_distance

    w, h = 480, 640
    # Chiếc ghế cách ~2.7m trên camera dọc (cao ~164px ~ 25.6% khung hình, chân y2=450)
    far_chair_box = {"x1": 165, "y1": 286, "x2": 315, "y2": 450}
    pin_d = estimate_distance_m("chair", far_chair_box, w, h)
    assert pin_d >= 2.5

    # Ngay cả khi Depth V2 đoán non còn 1.8m -> sau hòa giải vẫn >= 2.4m (vượt ngưỡng nguy hiểm 2.25m)
    calibrated = resolve_calibrated_distance("chair", metric_dist=1.8, pinhole_dist=pin_d, box=far_chair_box, img_w=w, img_h=h)
    assert calibrated >= 2.4

    # Còn nếu ghế thật sự đứng sát ở 1.2m (cao 365px, chạm gần đáy ảnh) -> phải giữ khoảng cách gần để báo ngay
    near_chair_box = {"x1": 90, "y1": 245, "x2": 390, "y2": 610}
    pin_near = estimate_distance_m("chair", near_chair_box, w, h)
    cal_near = resolve_calibrated_distance("chair", metric_dist=1.2, pinhole_dist=pin_near, box=near_chair_box, img_w=w, img_h=h)
    assert cal_near <= 1.4


def test_images_folder_evaluator_helpers():
    import evaluate_images_folder as ev

    assert ev.parse_true_distance_m("ghe_2.8m.jpg") == 2.8
    assert ev.parse_true_distance_m("cauthang_1,5m.png") == 1.5
    assert ev.parse_true_distance_m("sample.jpg") is None

    verdict, _ = ev.evaluate_category(
        "vat_can_xa",
        {"objects": [{"class_name": "chair", "distance_m": 2.7, "is_danger": False}], "navigation": {}},
        true_dist_m=2.8,
    )
    assert verdict == "PASS"


def test_depth_service_pothole_mirror_and_floor_filtering():
    """Kiểm tra lọc ổ gà theo SegFormer, đo khoảng cách mép gần ổ gà, phát hiện gương và chặn báo nhầm nền gạch."""
    import threading
    from types import SimpleNamespace
    import numpy as np
    from app.vision.depth_service import DepthService
    from app.vision.decision_engine import get_vietnamese_name

    svc = DepthService.__new__(DepthService)
    svc._lock = threading.Lock()
    svc.is_loaded = True
    svc.seg_loaded = True
    svc.pothole_model = None
    svc._latest_depth_map = None
    svc._latest_seg_hazards_norm = []
    svc._latest_potholes = []
    # ADE20K ids: 3=floor, 53=stairs, 27=mirror, 91=earth, 0=wall
    svc.seg_model = SimpleNamespace(
        config=SimpleNamespace(id2label={0: "wall", 3: "floor", 21: "water", 27: "mirror", 53: "stairs", 91: "earth"})
    )

    h, w = 480, 640
    # 1. Đo khoảng cách ổ gà ở mép gần phía dưới (tránh bị nền đường 4.5m ở nửa trên hộp kéo xa)
    dmap = np.linspace(5.0, 1.4, h, dtype=np.float32)[:, None] * np.ones((1, w), dtype=np.float32)
    ph_dist = svc._estimate_pothole_distance(10, int(h * 0.30), 630, h, dmap, w, h)
    assert ph_dist <= 2.0

    # 2. Lọc hộp ổ gà giả nằm trên cầu thang (seg=stairs) vs giữ ổ gà trên đường đất (seg=earth)
    seg_stairs = np.full((64, 64), 53, dtype=np.int32)
    assert svc._is_valid_pothole_box(10, int(h * 0.50), 600, h, w, h, seg_stairs) is False
    seg_earth = np.full((64, 64), 91, dtype=np.int32)
    assert svc._is_valid_pothole_box(10, int(h * 0.50), 600, h, w, h, seg_earth) is True

    # 3. Chặn báo giả cầu thang khi vùng hộp thực chất toàn nền gạch phẳng (seg=floor)
    svc._latest_seg_map = np.full((64, 64), 3, dtype=np.int32)
    fake_stairs = {"class_name": "stairs", "bounding_box": {"x1": 60, "y1": 220, "x2": 580, "y2": 450}}
    assert svc._is_stairs_suppressed_by_floor_seg(fake_stairs, w, h) is True

    # 4. Nhận diện tấm gương lớn dù bên trong gương DepthV2 đo sâu tới 5.5m
    seg_mirror = np.zeros((64, 64), dtype=np.int32)  # 0=wall
    seg_mirror[8:58, 14:50] = 27  # mirror (~43% khung hình)
    depth_small = np.full((64, 64), 1.6, dtype=np.float32)
    depth_small[10:56, 16:48] = 5.5  # phản chiếu phòng sâu bên trong gương
    hazards = svc._extract_seg_hazards_normalized(seg_mirror, depth_small)
    mirror_objs = [o for o in hazards if o["class_name"] == "mirror"]
    assert len(mirror_objs) == 1
    assert mirror_objs[0]["distance_m"] <= 1.7
    assert mirror_objs[0]["is_danger"] is True
    assert get_vietnamese_name("mirror") == "tấm gương"


def test_step1_yoloe_aliases_mirror_ring_and_training_remap():
    """Kiểm tra chuẩn hóa nhãn YOLOE/ObstacleYOLO, lọc vật ảo trong gương, đo viền gương và bảng ánh xạ dataset train."""
    from types import SimpleNamespace
    import numpy as np
    from app.vision.pipeline import _normalize_yolo_class, _filter_reflections_inside_mirrors
    from app.vision.depth_service import DepthService
    from app.vision.decision_engine import get_vietnamese_name
    from training.prepare_and_train_yolo import map_label_to_custom_class

    assert _normalize_yolo_class("flight of stairs") == "stairs"
    assert _normalize_yolo_class("water puddle") == "pothole"
    assert _normalize_yolo_class("open manhole") == "open_manhole"
    assert _normalize_yolo_class("glass door") == "glass_door"
    assert _normalize_yolo_class("bump") == "step"
    assert _normalize_yolo_class("pole") == "column"
    assert get_vietnamese_name("glass_door") == "cửa kính"
    assert get_vietnamese_name("open_manhole") == "hố ga hở"

    # Vật ảo nằm trong khung gương bị lọc, vật đứng ngoài/trước gương vẫn được giữ lại
    objs = [
        {"class_name": "mirror", "bounding_box": {"x1": 60, "y1": 10, "x2": 580, "y2": 460}},
        {"class_name": "chair", "bounding_box": {"x1": 150, "y1": 220, "x2": 310, "y2": 390}},  # lọt trong gương
        {"class_name": "chair", "bounding_box": {"x1": 0, "y1": 260, "x2": 55, "y2": 450}},     # ở bên ngoài gương
    ]
    filtered = _filter_reflections_inside_mirrors(objs)
    assert len(filtered) == 2
    assert [o["class_name"] for o in filtered] == ["mirror", "chair"]
    assert filtered[1]["bounding_box"]["x1"] == 0

    # Đo độ sâu mép viền gương lớn không bị lòng gương 6.0m đánh lừa
    svc = DepthService.__new__(DepthService)
    dmap = np.full((480, 640), 1.5, dtype=np.float32)
    dmap[40:440, 100:540] = 6.0
    ring_d = svc.estimate_surface_ring_depth(80, 20, 560, 450, dmap, 640, 480)
    assert ring_d <= 1.7

    # Mapping dataset 25 lớp -> 15 lớp chuẩn train
    assert map_label_to_custom_class("Electrical Pole") == "column"
    assert map_label_to_custom_class("Manhole") == "open_manhole"
    assert map_label_to_custom_class("Stairs") == "stairs"
    assert map_label_to_custom_class("Motorcycle") == "vehicle"



