import pytest
import numpy as np
from app.vision.geometric_hazard_detector import GeometricHazardDetector


@pytest.fixture
def detector():
    return GeometricHazardDetector()


def test_detect_wall_on_flat_vertical_surface(detector):
    """Bức tường phẳng thẳng đứng ở vùng trung tâm (dZ/dy ~ 0) phải được phát hiện."""
    h, w = 480, 640
    dmap = np.zeros((h, w), dtype=np.float32)

    # Nửa dưới là sàn nhà dốc lên
    for y in range(int(h * 0.65), h):
        ratio = (h - y) / (h * 0.35)
        dmap[y, :] = 0.5 + (1.0 - ratio) * 0.9

    # Nửa trên là bức tường phẳng 1.3m
    dmap[:int(h * 0.65), :] = 1.3

    res = detector.detect_wall(dmap)
    assert res is not None
    assert res["class_name"] == "wall"
    assert res["class_name_vn"] == "bức tường"
    assert 1.1 <= res["distance_m"] <= 1.5
    assert res["is_danger"] is True


def test_detect_wall_ignores_open_road(detector):
    """Mặt đường phẳng thông thoáng (dZ/dy tăng đều đặn) không được báo nhầm là tường."""
    h, w = 480, 640
    # Đường phẳng dốc đều từ 0.8m lên 8.0m
    dmap = np.linspace(0.8, 8.0, h, dtype=np.float32)[:, None] * np.ones((1, w), dtype=np.float32)

    res = detector.detect_wall(dmap)
    assert res is None, "Mặt đường thông thoáng không được báo nhầm là bức tường"


def test_detect_stairs_horizontal_edges_with_depth_jumps(detector):
    """Bậc thang với các vệt ngang và bước nhảy độ sâu phải được nhận diện."""
    h, w = 480, 640
    import cv2
    frame = np.full((h, w, 3), 190, dtype=np.uint8)
    dmap = np.zeros((h, w), dtype=np.float32)

    # 3 bậc ngang rõ rệt
    step_ys = [int(h * 0.55), int(h * 0.68), int(h * 0.80)]
    for y in step_ys:
        cv2.line(frame, (int(w * 0.15), y), (int(w * 0.85), y), (20, 20, 20), 4)

    # Độ sâu nhảy bậc
    cur_z = 2.4
    prev_y = 0
    for y in step_ys:
        dmap[prev_y:y, :] = cur_z
        cur_z -= 0.4
        prev_y = y
    dmap[prev_y:, :] = 0.8

    res = detector.detect_stairs_and_curbs(frame, dmap)
    assert res is not None
    assert res["class_name"] in ("stairs", "step")
    assert res["distance_m"] > 0
    assert res["is_danger"] is True


def test_detect_single_step_doorstep(detector):
    """Một bậc đơn lẻ (gờ cửa / bậc thềm) có bước nhảy độ sâu phải nhận diện là 'bậc thềm'."""
    h, w = 480, 640
    import cv2
    frame = np.full((h, w, 3), 190, dtype=np.uint8)
    dmap = np.zeros((h, w), dtype=np.float32)

    # 1 gờ ngang bậc thềm ở y = 70%
    step_y = int(h * 0.70)
    cv2.line(frame, (int(w * 0.15), step_y), (int(w * 0.85), step_y), (10, 10, 10), 5)

    # Trước bậc thềm là 1.8m, sau bậc thềm là 1.0m (chênh lệch 0.8m)
    dmap[:step_y, :] = 1.8
    dmap[step_y:, :] = 1.0

    res = detector.detect_stairs_and_curbs(frame, dmap)
    assert res is not None
    assert res["class_name_vn"] == "bậc thềm"
    assert res["num_steps"] == 1
    assert res["is_danger"] is True


def test_detect_optical_closeup_wall_without_depth_map(detector):
    """Khi quay sát vào bức tường phẳng (ngay cả khi dmap=None), lớp quang học vẫn phát hiện được tường."""
    h, w = 480, 640
    # Bức tường sơn màu kem đồng nhất
    frame = np.full((h, w, 3), (210, 222, 228), dtype=np.uint8)
    res = detector.detect_wall(None, frame_bgr=frame, known_boxes=[])
    assert res is not None
    assert res["class_name"] == "wall"
    assert res["class_name_vn"] == "bức tường"
    assert res["is_danger"] is True


def test_flat_wall_never_triggers_false_stairs(detector):
    """Quay vào bức tường đứng (depth ~ 1.2m đồng nhất) dù có vệt ngang công tắc điện cũng KHÔNG ĐƯỢC báo nhầm là cầu thang."""
    import cv2
    h, w = 480, 640
    frame = np.full((h, w, 3), (180, 200, 185), dtype=np.uint8)
    # Vẽ vệt ngang giả lập công tắc/gờ sơn trên tường
    cv2.line(frame, (int(w * 0.2), int(h * 0.55)), (int(w * 0.8), int(h * 0.55)), (80, 80, 80), 3)
    cv2.line(frame, (int(w * 0.2), int(h * 0.65)), (int(w * 0.8), int(h * 0.65)), (80, 80, 80), 3)
    dmap = np.full((h, w), 1.2, dtype=np.float32)

    stairs_res = detector.detect_stairs_and_curbs(frame, dmap, known_boxes=[])
    assert stairs_res is None, "Bức tường đứng không bao giờ được báo nhầm là cầu thang!"


def test_edgesam_mask_refined_depth_and_unlabeled_obstacle():
    """EdgeSAM hỗ trợ đo độ sâu chính xác theo mặt nạ pixel (loại bỏ nền xa) và phát hiện vật cản chưa có nhãn YOLO."""
    import time
    from app.vision.sam_service import SamService

    svc = SamService()
    h, w = 480, 640
    # Giả lập nền xa 4.5m nhưng có vật cản ở trung tâm cách 1.2m
    dmap = np.full((h, w), 4.5, dtype=np.float32)
    dmap[150:380, 220:420] = 1.2

    # Mặt nạ EdgeSAM ôm khít vật cản ở vùng [220..420, 150..380]
    svc._latest_masks = [
        {
            "area_px": 46000,
            "area_ratio": 0.25,
            "bounding_box": {"x1": 220, "y1": 150, "x2": 420, "y2": 380},
            "polygon": [[220, 150], [420, 150], [420, 380], [220, 380]],
        }
    ]
    svc._latest_masks_time = time.time()
    svc._latest_img_size = (w, h)

    # 1. Đo khoảng cách tinh chỉnh theo mặt nạ EdgeSAM cho hộp YOLO bị rộng
    yolo_box = {"x1": 180, "y1": 120, "x2": 460, "y2": 410}
    refined_d = svc.get_mask_refined_depth(yolo_box, dmap)
    assert refined_d == 1.2

    # 2. Phát hiện vật cản lớn chắn giữa lối đi khi YOLO không bắt được hộp nào
    unlabeled = svc.get_unlabeled_mask_obstacles([], dmap, w, h)
    assert len(unlabeled) == 1
    assert unlabeled[0]["class_name_vn"] == "vật cản"
    assert unlabeled[0]["distance_m"] == 1.2
    assert unlabeled[0]["is_danger"] is True


def test_column_with_side_gap_tells_user_to_turn_right():
    """Cột ở giữa, bên phải khung hình còn trống -> báo ngay rẽ sang phải."""
    from app.vision.guided_scan import reset_voice_state, update_spatial_scan_and_voice, voice_state

    reset_voice_state()
    objs = [{"class_name": "pillar", "class_name_vn": "cột", "position": "center",
             "distance_m": 1.6, "bounding_box": {"x1": 200, "y1": 0, "x2": 360, "y2": 480}}]
    lanes = {"lanes": {"left": {"median_m": 1.1, "is_blocked": True},
                       "center": {"median_m": 1.6, "is_blocked": False},
                       "right": {"median_m": 3.5, "is_blocked": False}}}
    # Frame 1: phát hiện cản và báo ngay lập tức
    text, _, speak = update_spatial_scan_and_voice("Phía trước có cột.", "HIGH", objs, lanes, 640, 100.0)
    assert speak is True
    assert "Phía trước có cột, hãy rẽ sang phải." in text

    # Frame tiếp theo cùng cảnh (sau 0.3s) -> giữ yên lặng, không lặp lại ồn ào
    _, _, speak2 = update_spatial_scan_and_voice("Phía trước có cột.", "HIGH", objs, lanes, 640, 100.3)
    assert speak2 is False


def test_far_wall_is_safe_and_close_wall_tells_turn_around():
    """Tường ở xa 2.0m thì an toàn đi tiếp; chỉ khi lại gần <= 1.3m chắn cả 2 bên mới báo quay đầu."""
    from app.vision.guided_scan import reset_voice_state, update_spatial_scan_and_voice

    reset_voice_state()
    t = 10.0

    # 1. Tường ở xa 2.0m: Chưa nguy hiểm, người khiếm thị vẫn đi thẳng an toàn
    wall_far = [{"class_name": "wall", "class_name_vn": "bức tường", "position": "center",
                 "distance_m": 2.0, "bounding_box": {"x1": 50, "y1": 50, "x2": 590, "y2": 430}}]
    lanes_far = {"lanes": {k: {"median_m": 2.0, "is_blocked": False} for k in ("left", "center", "right")}}

    update_spatial_scan_and_voice("Phía trước có bức tường.", "LOW", wall_far, lanes_far, 640, t)
    t += 0.2
    text1, _, speak1 = update_spatial_scan_and_voice("Phía trước có bức tường.", "LOW", wall_far, lanes_far, 640, t)
    assert speak1 is True
    assert "Phía trước an toàn, đi thẳng." in text1

    # 2. Khi lại gần <= 1.0m chắn kín cả 2 bên: Báo quay đầu dứt khoát ngay lập tức (frame 1)
    t += 2.0
    wall_close = [{"class_name": "wall", "class_name_vn": "bức tường", "position": "center",
                  "distance_m": 1.0, "bounding_box": {"x1": 0, "y1": 0, "x2": 640, "y2": 480}}]
    lanes_close = {"lanes": {k: {"median_m": 1.0, "is_blocked": True} for k in ("left", "center", "right")}}

    text2, _, speak2 = update_spatial_scan_and_voice("Phía trước có bức tường.", "HIGH", wall_close, lanes_close, 640, t)
    assert speak2 is True
    assert "cả hai bên đều vướng, hãy quay đầu." in text2

    # Frame tiếp theo sau 0.2s: Giữ yên lặng chống nói lặp
    t += 0.2
    _, _, speak2_repeat = update_spatial_scan_and_voice("Phía trước có bức tường.", "HIGH", wall_close, lanes_close, 640, t)
    assert speak2_repeat is False


def test_chair_in_front_warns_turn_immediately():
    """Chiếc ghế ở ngay trước mặt (dù cách 2.2m hoặc hơi lệch làn) BẮT BUỘC cảnh báo rẽ, tuyệt đối không báo an toàn."""
    from app.vision.guided_scan import reset_voice_state, update_spatial_scan_and_voice, assess_path

    reset_voice_state()
    # 1. Chiếc ghế ở làn giữa cách 2.2m: Đường trước mặt bị chắn
    chair_center = [{
        "class_name": "chair",
        "class_name_vn": "ghế",
        "position": "center",
        "distance_m": 2.2,
        "in_roi": True,
        "bounding_box": {"x1": 200, "y1": 150, "x2": 440, "y2": 450}
    }]
    path_res = assess_path("Phía trước 2 mét có ghế.", chair_center, None, 640)
    assert path_res["status"] in ("TURN_LEFT", "TURN_RIGHT", "TURN_AROUND")
    assert path_res["name"] == "ghế"
    assert path_res["status"] != "CLEAR"

    # Frame 1: Cảnh báo phát ra ngay lập tức
    text, prio, speak = update_spatial_scan_and_voice("Phía trước có ghế.", "HIGH", chair_center, None, 640, 50.0)
    assert speak is True
    assert "Phía trước có ghế" in text
    assert ("rẽ sang" in text or "quay đầu" in text)
    assert "an toàn" not in text

    # 2. Chiếc ghế ở phía trước lệch nhẹ sang trái (x1=80, x2=340) cách 1.8m:
    # Bắt buộc phát hiện xâm lấn làn giữa và chỉ dẫn rẽ sang phải
    chair_leftish = [{
        "class_name": "chair",
        "class_name_vn": "ghế",
        "position": "left",
        "distance_m": 1.8,
        "in_roi": True,
        "bounding_box": {"x1": 80, "y1": 150, "x2": 340, "y2": 450}
    }]
    reset_voice_state()
    res_leftish = assess_path("Phía trước có ghế.", chair_leftish, None, 640)
    assert res_leftish["status"] == "TURN_RIGHT"
    assert res_leftish["name"] == "ghế"

    text_r, _, speak_r = update_spatial_scan_and_voice("Phía trước có ghế.", "HIGH", chair_leftish, None, 640, 60.0)
    assert speak_r is True
    assert "Phía trước có ghế, hãy rẽ sang phải." in text_r


def test_ceramic_tile_floor_grout_lines_not_falsely_detected_as_stairs(detector):
    """Nền gạch men phẳng có các đường ron ngang/dọc và độ sâu giảm mượt theo mặt sàn tuyệt đối KHÔNG báo là cầu thang."""
    import cv2
    h, w = 480, 640
    frame = np.full((h, w, 3), 210, dtype=np.uint8)

    # Vẽ các đường ron gạch ngang và dọc của nền gạch men
    for y in [int(h * 0.48), int(h * 0.62), int(h * 0.76), int(h * 0.88)]:
        cv2.line(frame, (int(w * 0.10), y), (int(w * 0.90), y), (60, 60, 60), 2)
    for x in [int(w * 0.25), int(w * 0.50), int(w * 0.75)]:
        cv2.line(frame, (x, int(h * 0.40)), (x, int(h * 0.95)), (60, 60, 60), 2)

    # Độ sâu sàn phẳng dốc đều từ 3.2m (phía xa) về 0.8m (dưới chân), không có bước nhảy bậc đột ngột
    dmap = np.linspace(3.2, 0.8, h, dtype=np.float32)[:, None] * np.ones((1, w), dtype=np.float32)

    res = detector.detect_stairs_and_curbs(frame, dmap)
    assert res is None, "Nền gạch men có rãnh không được phép nhận nhầm thành cầu thang/bậc thềm"


def test_cabinet_with_bottom_shelves_not_falsely_detected_as_stairs(detector):
    """Chiếc tủ có các thanh kệ ngang ở dưới (mặt đứng đồng nhất ~1.4m) tuyệt đối KHÔNG báo là cầu thang."""
    import cv2
    h, w = 480, 640
    frame = np.full((h, w, 3), 180, dtype=np.uint8)

    # Vẽ khung tủ dọc và 3 thanh kệ ngang bên dưới
    cv2.rectangle(frame, (int(w * 0.20), int(h * 0.20)), (int(w * 0.80), int(h * 0.92)), (40, 40, 40), 4)
    for y in [int(h * 0.52), int(h * 0.68), int(h * 0.84)]:
        cv2.line(frame, (int(w * 0.20), y), (int(w * 0.80), y), (30, 30, 30), 4)

    # Độ sâu chiếc tủ là mặt phẳng đứng ~1.4m (chênh lệch nhẹ ở các hốc kệ)
    dmap = np.full((h, w), 1.4, dtype=np.float32)
    dmap[int(h * 0.52):int(h * 0.68), :] = 1.50
    dmap[int(h * 0.68):int(h * 0.84), :] = 1.42

    res = detector.detect_stairs_and_curbs(frame, dmap)
    assert res is None, "Chiếc tủ có thanh kệ ở dưới không được nhận nhầm thành cầu thang"


def test_far_chair_5_5m_never_triggers_turn_warning():
    """Chiếc ghế nằm ở xa 5.5m (hoặc > 2.25m) tuyệt đối KHÔNG được báo nguy hiểm rẽ phải/rẽ trái mà phải báo đi thẳng an toàn."""
    from app.vision.guided_scan import reset_voice_state, update_spatial_scan_and_voice, assess_path

    reset_voice_state()
    far_chair = [{
        "class_name": "chair",
        "class_name_vn": "ghế",
        "position": "center",
        "distance_m": 5.5,
        "in_roi": True,
        "bounding_box": {"x1": 290, "y1": 180, "x2": 350, "y2": 240}
    }]
    res = assess_path("Phía trước cách 5.5 mét có ghế.", far_chair, None, 640)
    assert res["status"] == "CLEAR"

    t = 100.0
    update_spatial_scan_and_voice("Phía trước cách 5.5 mét có ghế.", "LOW", far_chair, None, 640, t)
    t += 0.2
    text, prio, speak = update_spatial_scan_and_voice("Phía trước cách 5.5 mét có ghế.", "LOW", far_chair, None, 640, t)
    assert speak is True
    assert text == "Phía trước an toàn, đi thẳng."
    assert prio == "LOW"


def test_detect_pillar_at_1_2m_in_front_even_with_far_chairs(detector):
    """Cây cột ngay trước mặt cách 1.2m BẮT BUỘC được phát hiện là 'cột nhà' và cảnh báo rẽ, kể cả khi ở xa 5m có ghế."""
    from app.vision.guided_scan import reset_voice_state, update_spatial_scan_and_voice

    h, w = 480, 640
    # Nền phòng học phía sau xa 4.2m, riêng cây cột đứng ở giữa (x: 200 -> 390) cách 1.2m
    dmap = np.full((h, w), 4.2, dtype=np.float32)
    dmap[:, 200:390] = 1.2

    pillar = detector.detect_pillar(dmap, known_boxes=[])
    assert pillar is not None
    assert pillar["class_name"] == "column"
    assert pillar["class_name_vn"] == "cột nhà"
    assert pillar["distance_m"] == 1.2
    assert pillar["is_danger"] is True

    # Kết hợp cây cột 1.2m với 2 chiếc ghế ở xa 5.2m ở hậu cảnh -> Vẫn phải cảnh báo né cây cột ngay lập tức!
    far_chairs = [
        {"class_name": "chair", "class_name_vn": "ghế", "position": "left", "distance_m": 5.2, "bounding_box": {"x1": 30, "y1": 200, "x2": 100, "y2": 280}},
        {"class_name": "chair", "class_name_vn": "ghế", "position": "right", "distance_m": 5.5, "bounding_box": {"x1": 500, "y1": 200, "x2": 580, "y2": 280}},
        pillar,
    ]
    reset_voice_state()
    text, prio, speak = update_spatial_scan_and_voice("Phía trước có cột nhà.", "HIGH", far_chairs, None, 640, 200.0)
    assert speak is True
    assert "Phía trước có cột nhà" in text
    assert "rẽ sang phải" in text or "rẽ sang trái" in text
    assert prio == "HIGH"


def test_multi_model_fusion_engine_deduplicates_and_synthesizes_guidance():
    """MultiModelFusionEngine phải khử trùng lặp khi YOLO bắt được 'ghế' còn Geometric/EdgeSAM báo đè 'wall'/'obstacle' và đưa ra quyết định rẽ tức thì."""
    from app.vision.fusion_engine import MultiModelFusionEngine

    engine = MultiModelFusionEngine()
    yolo_objs = [{
        "class_name": "chair",
        "class_name_vn": "ghế",
        "confidence": 0.85,
        "position": "center",
        "distance_m": 1.4,
        "in_roi": True,
        "is_danger": True,
        "bounding_box": {"x1": 210, "y1": 140, "x2": 430, "y2": 440},
    }]
    structural_hazards = [{
        "class_name": "wall",
        "class_name_vn": "bức tường",
        "confidence": 0.80,
        "position": "center",
        "distance_m": 1.5,
        "in_roi": True,
        "is_danger": True,
        "bounding_box": {"x1": 100, "y1": 80, "x2": 540, "y2": 450},
    }]
    sam_obstacles = [{
        "class_name": "obstacle",
        "class_name_vn": "vật cản",
        "confidence": 0.78,
        "position": "center",
        "distance_m": 1.4,
        "in_roi": True,
        "is_danger": True,
        "bounding_box": {"x1": 215, "y1": 145, "x2": 425, "y2": 435},
    }]
    depth_lanes = {
        "has_depth": True,
        "lanes": {
            "left": {"median_m": 3.5, "is_blocked": False},
            "center": {"median_m": 1.4, "is_blocked": True},
            "right": {"median_m": 2.8, "is_blocked": False},
        },
    }

    res = engine.fuse_detections(yolo_objs, structural_hazards, sam_obstacles, depth_lanes, 640, 480, now=300.0)
    fused = res["fused_objects"]
    summary = res["fusion_summary"]

    # Nhãn 'wall' và 'obstacle' trùng vùng với 'chair' phải bị loại bỏ, chỉ giữ lại 'chair'
    assert len(fused) == 1
    assert fused[0]["class_name"] == "chair"
    assert summary["instant_action"] == "TURN_LEFT"
    assert summary["lane_clearance"]["center"]["primary_hazard"] == "ghế"


def test_blind_user_worst_case_safety_interrupts_and_dark_room(detector):
    """Kiểm thử các tình huống khắc nghiệt (Worst-Case) từ góc nhìn người khiếm thị:
    1. Ngắt khẩn cấp 0ms khi vừa báo 'An toàn' được 0.25s thì có người/cầu thang cắt ngang.
    2. Ưu tiên cảnh báo cầu thang trong all_objects kể cả khi câu text bị vật 2 bên che lấp.
    3. Cảnh báo lặp khẩn cấp khi người dùng tiến sát < 0.85m chỉ sau 2.2s.
    4. Không báo nhầm bóng tối / che camera thành bức tường.
    """
    from app.vision.guided_scan import reset_voice_state, update_spatial_scan_and_voice

    W, H = 640, 480
    reset_voice_state()
    update_spatial_scan_and_voice("", "LOW", [], None, W, 10.0)
    s_clear, _, sp_clear = update_spatial_scan_and_voice("", "LOW", [], None, W, 12.0)
    assert sp_clear is True and s_clear == "Phía trước an toàn, đi thẳng."

    # Chỉ 0.25s sau khi nghe "An toàn", có người bước ra cách 1.1m -> Phải ngắt và cảnh báo NGAY!
    sudden_person = [{
        "class_name": "person", "class_name_vn": "người", "position": "center",
        "distance_m": 1.1, "in_roi": True, "is_danger": True,
        "bounding_box": {"x1": 220, "y1": 80, "x2": 420, "y2": 460},
    }]
    s_interrupt, _, sp_interrupt = update_spatial_scan_and_voice("Phía trước có người.", "HIGH", sudden_person, None, W, 12.25)
    assert sp_interrupt is True
    assert "Phía trước có người" in s_interrupt

    # Tiến sát nguy kịch từ 2.0m xuống 0.65m sau 2.2s -> Phải nhắc lại ngay không chờ hết 6s
    reset_voice_state()
    chair_2m = [{"class_name": "chair", "class_name_vn": "ghế", "position": "center", "distance_m": 2.0, "in_roi": True, "is_danger": True, "bounding_box": {"x1": 240, "y1": 160, "x2": 400, "y2": 420}}]
    update_spatial_scan_and_voice("Phía trước có ghế.", "HIGH", chair_2m, None, W, 20.0)
    chair_065m = [{"class_name": "chair", "class_name_vn": "ghế", "position": "center", "distance_m": 0.65, "in_roi": True, "is_danger": True, "bounding_box": {"x1": 120, "y1": 80, "x2": 520, "y2": 470}}]
    s_crit, _, sp_crit = update_spatial_scan_and_voice("Phía trước có ghế.", "HIGH", chair_065m, None, W, 22.2)
    assert sp_crit is True

    # Vùng tối om (intensity = 12) không được báo giả bức tường
    dark_frame = np.full((H, W, 3), 12, dtype=np.uint8)
    assert detector.detect_wall(None, known_boxes=[], frame_bgr=dark_frame) is None


def test_fusion_engine_model_branch_breakdown_and_sam_masks():
    from app.vision.fusion_engine import MultiModelFusionEngine

    engine = MultiModelFusionEngine()
    yolo_objs = [
        {
            "track_id": 1,
            "class_name": "chair",
            "vi_name": "ghế",
            "confidence": 0.91,
            "bbox": [240, 180, 400, 420],
            "distance_m": 1.6,
            "risk_level": "HIGH",
            "sam_mask_refined": True,
        }
    ]
    structural_hazards = [
        {
            "class_name": "staircase",
            "vi_name": "cầu thang",
            "confidence": 0.85,
            "bbox": [480, 220, 620, 440],
            "distance_m": 1.8,
            "risk_level": "HIGH",
            "source_model": "SegFormer+Geo3D",
        }
    ]
    sam_obstacles = [
        {
            "class_name": "obstacle",
            "vi_name": "vật cản lạ",
            "confidence": 0.72,
            "bbox": [20, 200, 120, 380],
            "distance_m": 2.4,
            "risk_level": "MEDIUM",
            "source_model": "EdgeSAM",
        }
    ]
    depth_lanes = {
        "lanes": {
            "left": {"median_m": 2.4},
            "center": {"median_m": 1.6},
            "right": {"median_m": 1.8},
        }
    }

    res = engine.fuse_detections(
        yolo_objs,
        structural_hazards,
        sam_obstacles,
        depth_lanes,
        640,
        480,
        now=500.0,
        sam_mask_count=4,
    )
    summary = res["fusion_summary"]
    branches = summary["model_branches"]

    assert {"yolo11", "depth_v2", "edgesam", "segformer_geo"}.issubset(branches.keys())
    assert branches["yolo11"]["count"] == 1
    assert branches["depth_v2"]["has_depth"] is True
    assert branches["depth_v2"]["center_m"] == 1.6
    assert branches["edgesam"]["active_masks"] == 4
    assert branches["edgesam"]["refined_boxes"] == 1
    assert branches["segformer_geo"]["count"] == 1
    assert summary["danger_obstacle_count"] >= 1
    assert summary["ema_active_tracks"] >= 1
    assert "dedup_removed_count" in summary


def test_decision_telemetry_and_navigation_sync():
    from app.vision.guided_scan import get_decision_telemetry, voice_state
    from app.vision.pipeline import _sync_navigation_with_decision

    snapshot = dict(voice_state)
    try:
        voice_state["current_status"] = "CLEAR"
        voice_state["clear_frames"] = 5
        voice_state["blocked_frames"] = 0
        voice_state["last_time"] = 123.4567
        voice_state["last_spoken_flag"] = True

        dt = get_decision_telemetry(now=125.4567)
        assert dt["state_code"] == "CLEAR"
        assert "CLEAR" in dt["state_label"]
        assert dt["clear_frames"] == 5
        assert dt["blocked_frames"] == 0
        assert dt["since_last_speech_s"] == 2.0
        assert dt["last_speech_ts"] == 123.457
        assert dt["spoken_this_frame"] is True
        assert "Blaze AI" in dt["voice_engine"]

        nav = {
            "direction": "right",
            "text": "Rẽ phải",
            "description": "Bên trái có vật cản, rẽ phải.",
        }
        synced = _sync_navigation_with_decision(nav, dt, "Phía trước an toàn, đi thẳng.")
        assert synced["direction"] == "straight"
        assert synced["text"] == "Đi thẳng"
        assert synced["description"] == "Phía trước an toàn, đi thẳng."

        synced_left = _sync_navigation_with_decision(dict(nav), {"state_code": "TURN_LEFT"}, "Rẽ trái tránh vật cản.")
        assert synced_left["direction"] == "left"
        assert synced_left["text"] == "Rẽ trái"

        synced_steps = _sync_navigation_with_decision(dict(nav), {"state_code": "STEPS"}, "Chú ý bậc cầu thang!")
        assert synced_steps["direction"] == "stop"
    finally:
        voice_state.clear()
        voice_state.update(snapshot)


def test_view_phone_stream_contains_fusion_hud_and_trace_log():
    from pathlib import Path

    html_path = Path(__file__).resolve().parents[1] / "view_phone_stream.html"
    content = html_path.read_text(encoding="utf-8")
    for token in (
        "fusionEngineSection",
        "fusionTraceList",
        "renderFusionEngineHud",
        "appendFusionTraceLog",
    ):
        assert token in content







