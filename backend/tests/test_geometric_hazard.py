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



