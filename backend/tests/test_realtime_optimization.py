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
