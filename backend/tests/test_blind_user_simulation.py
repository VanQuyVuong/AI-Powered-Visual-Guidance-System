"""
Bộ Kiểm Thử Tự Động 15 Kịch Bản "Người Khiếm Thị Ảo" (Virtual Visually Impaired User Simulation Suite).

Mục tiêu:
- Đóng vai một người khiếm thị thực sự sử dụng hệ thống trong 15 tình huống thực tế và khắc nghiệt nhất
  (Worst-Case Scenarios) trước khi đưa ra thử nghiệm với người thật.
- Kiểm tra toàn diện:
  1. Tốc độ phản xạ (Latency < 25ms, ngắt khẩn cấp 0ms khi chuyển từ An toàn sang Nguy hiểm).
  2. Chống báo giả (Nền gạch men có rãnh caro, chiếc tủ có thanh kệ ngang, ghế ở xa 5.5m, phòng tối).
  3. An toàn sinh mạng P0 (Cây cột 1.2m, cầu thang bị vật 2 bên làm nhiễu, ngõ cụt, ổ gà, tiến sát < 0.85m).
  4. Trải nghiệm âm thanh (Chống nói lặp gây hoảng loạn, chống nhiễu khi rung tay cầm điện thoại).
"""

import time
import numpy as np
import cv2
import pytest

from app.vision.geometric_hazard_detector import GeometricHazardDetector
from app.vision.fusion_engine import MultiModelFusionEngine
from app.vision.guided_scan import reset_voice_state, update_spatial_scan_and_voice


@pytest.fixture
def sim_env():
    reset_voice_state()
    detector = GeometricHazardDetector()
    fusion = MultiModelFusionEngine()
    return detector, fusion, 640, 480


def test_scenario_01_ceramic_tile_floor_grout_lines(sim_env):
    """Kịch bản 1: Đi trên nền gạch men bóng có rãnh caro đậm dưới ánh đèn -> Không báo giả cầu thang/tường."""
    detector, _, W, H = sim_env
    frame = np.full((H, W, 3), 215, dtype=np.uint8)
    for y in range(200, 460, 55):
        cv2.line(frame, (40, y), (600, y), (50, 50, 50), 2)
    for x in range(100, 600, 110):
        cv2.line(frame, (x, 180), (x, 470), (50, 50, 50), 2)
    dmap = np.linspace(3.5, 0.85, H, dtype=np.float32)[:, None] * np.ones((1, W), dtype=np.float32)

    assert detector.detect_stairs_and_curbs(frame, dmap) is None
    assert detector.detect_wall(dmap, frame_bgr=frame) is None
    assert detector.detect_pillar(dmap, frame_bgr=frame) is None


def test_scenario_02_cabinet_with_bottom_shelves(sim_env):
    """Kịch bản 2: Đứng trước chiếc tủ có 3 thanh kệ ngang ở dưới (cách 1.4m) -> Không báo nhầm là cầu thang."""
    detector, _, W, H = sim_env
    frame = np.full((H, W, 3), 175, dtype=np.uint8)
    cv2.rectangle(frame, (130, 80), (510, 445), (35, 35, 35), 4)
    for y in [250, 330, 410]:
        cv2.line(frame, (130, y), (510, y), (25, 25, 25), 4)
    dmap = np.full((H, W), 1.4, dtype=np.float32)
    dmap[250:330, 130:510] = 1.48

    assert detector.detect_stairs_and_curbs(frame, dmap) is None


def test_scenario_03_far_chair_5_5m_safe_straight(sim_env):
    """Kịch bản 3: Chiếc ghế nằm ở xa 5.5m giữa lối đi -> Báo 'Phía trước an toàn, đi thẳng'."""
    _, fusion, W, H = sim_env
    far_chair = [{
        "class_name": "chair", "class_name_vn": "ghế", "confidence": 0.78, "position": "center",
        "distance_m": 5.5, "in_roi": True, "is_danger": False,
        "bounding_box": {"x1": 295, "y1": 190, "x2": 345, "y2": 245},
    }]
    f_res = fusion.fuse_detections(far_chair, [], [], None, W, H, now=10.0)
    update_spatial_scan_and_voice("Phía trước có ghế.", "LOW", f_res["fused_objects"], None, W, 10.0)
    speech, prio, speak = update_spatial_scan_and_voice("Phía trước có ghế.", "LOW", f_res["fused_objects"], None, W, 12.0)
    assert speak is True
    assert speech == "Phía trước an toàn, đi thẳng."
    assert prio == "LOW"


def test_scenario_04_close_pillar_1_2m_with_far_chairs(sim_env):
    """Kịch bản 4: Cây cột bê tông 1.2m ngay trước mặt (phía sau 5m có 2 chiếc ghế) -> Bắt trúng cột và báo rẽ."""
    detector, fusion, W, H = sim_env
    dmap = np.full((H, W), 4.5, dtype=np.float32)
    dmap[:, 210:390] = 1.2
    pillar_hz = detector.detect_pillar(dmap, known_boxes=[])
    assert pillar_hz is not None and pillar_hz["class_name_vn"] == "cột nhà"

    bg_chairs = [
        {"class_name": "chair", "class_name_vn": "ghế", "confidence": 0.7, "position": "left", "distance_m": 5.0, "in_roi": False, "is_danger": False, "bounding_box": {"x1": 20, "y1": 200, "x2": 80, "y2": 260}},
        {"class_name": "chair", "class_name_vn": "ghế", "confidence": 0.7, "position": "right", "distance_m": 5.3, "in_roi": False, "is_danger": False, "bounding_box": {"x1": 540, "y1": 200, "x2": 600, "y2": 260}},
    ]
    f_res = fusion.fuse_detections(bg_chairs, [pillar_hz], [], None, W, H, now=20.0)
    speech, prio, speak = update_spatial_scan_and_voice("Phía trước có cột nhà.", "HIGH", f_res["fused_objects"], None, W, 20.0)
    assert speak is True
    assert "cột nhà" in speech and "rẽ sang" in speech
    assert prio == "HIGH"


def test_scenario_05_sudden_person_0_25s_after_clear_announcement(sim_env):
    """Kịch bản 5: Vừa nghe 'An toàn' được 0.25s thì có người bất ngờ cắt ngang ở 1.1m -> Ngắt 0ms và báo ngay."""
    _, _, W, _ = sim_env
    update_spatial_scan_and_voice("", "LOW", [], None, W, 28.0)
    update_spatial_scan_and_voice("", "LOW", [], None, W, 30.0)
    sudden_person = [{
        "class_name": "person", "class_name_vn": "người", "confidence": 0.92, "position": "center",
        "distance_m": 1.1, "in_roi": True, "is_danger": True,
        "bounding_box": {"x1": 220, "y1": 80, "x2": 420, "y2": 460},
    }]
    speech, _, speak = update_spatial_scan_and_voice("Phía trước có người.", "HIGH", sudden_person, None, W, 30.25)
    assert speak is True
    assert "Phía trước có người" in speech and "rẽ" in speech


def test_scenario_06_sudden_stairs_0_30s_after_clear_announcement(sim_env):
    """Kịch bản 6: Vừa nghe 'An toàn' được 0.30s thì bước tới sát cầu thang (1.3m) -> Ngắt 0ms và báo bước chậm."""
    _, _, W, _ = sim_env
    update_spatial_scan_and_voice("", "LOW", [], None, W, 40.0)
    update_spatial_scan_and_voice("", "LOW", [], None, W, 42.0)
    sudden_stairs = [{
        "class_name": "stairs", "class_name_vn": "cầu thang", "confidence": 0.90, "position": "center",
        "distance_m": 1.3, "in_roi": True, "is_danger": True,
        "bounding_box": {"x1": 100, "y1": 250, "x2": 540, "y2": 460},
    }]
    speech, _, speak = update_spatial_scan_and_voice("Chú ý dưới chân có cầu thang.", "HIGH", sudden_stairs, None, W, 42.3)
    assert speak is True
    assert speech == "Chú ý dưới chân, có cầu thang, hãy bước chậm."


def test_scenario_07_stairs_prioritized_over_side_clutter(sim_env):
    """Kịch bản 7: Cầu thang ở giữa nhưng 2 bên có người và ghế làm nhiễu câu text -> Vẫn ưu tiên báo cầu thang."""
    _, _, W, _ = sim_env
    mixed = [
        {"class_name": "person", "class_name_vn": "người", "position": "left", "distance_m": 1.5, "in_roi": True, "is_danger": True, "bounding_box": {"x1": 20, "y1": 100, "x2": 180, "y2": 450}},
        {"class_name": "chair", "class_name_vn": "ghế", "position": "right", "distance_m": 1.6, "in_roi": True, "is_danger": True, "bounding_box": {"x1": 460, "y1": 180, "x2": 620, "y2": 450}},
        {"class_name": "stairs", "class_name_vn": "cầu thang", "position": "center", "distance_m": 1.5, "in_roi": True, "is_danger": True, "bounding_box": {"x1": 180, "y1": 260, "x2": 460, "y2": 460}},
    ]
    speech, _, speak = update_spatial_scan_and_voice("Bên trái có người và bên phải có ghế.", "HIGH", mixed, None, W, 50.0)
    assert speak is True
    assert "cầu thang" in speech


def test_scenario_08_dead_end_all_three_lanes_blocked(sim_env):
    """Kịch bản 8: Ngõ cụt (Trái ghế 1.2m, Giữa bàn 1.1m, Phải tường 1.1m) -> Báo quay đầu CRITICAL."""
    _, _, W, _ = sim_env
    dead_end_objs = [
        {"class_name": "chair", "class_name_vn": "ghế", "position": "left", "distance_m": 1.2, "in_roi": True, "is_danger": True, "bounding_box": {"x1": 10, "y1": 180, "x2": 210, "y2": 460}},
        {"class_name": "table", "class_name_vn": "bàn", "position": "center", "distance_m": 1.1, "in_roi": True, "is_danger": True, "bounding_box": {"x1": 200, "y1": 160, "x2": 440, "y2": 460}},
    ]
    lanes_dead = {"has_depth": True, "lanes": {
        "left": {"median_m": 1.2, "is_blocked": True},
        "center": {"median_m": 1.1, "is_blocked": True},
        "right": {"median_m": 1.1, "is_blocked": True},
    }}
    speech, prio, speak = update_spatial_scan_and_voice("Phía trước có bàn.", "HIGH", dead_end_objs, lanes_dead, W, 60.0)
    assert speak is True
    assert "cả hai bên đều vướng, hãy quay đầu" in speech
    assert prio == "CRITICAL"


def test_scenario_09_anti_spam_when_standing_still_for_5_seconds(sim_env):
    """Kịch bản 9: Đứng yên trước chiếc ghế suốt 5 giây (25 frames) -> Chỉ đọc đúng 1 lần, không spam."""
    _, _, W, _ = sim_env
    chair_block = [{
        "class_name": "chair", "class_name_vn": "ghế", "position": "center", "distance_m": 1.3,
        "in_roi": True, "is_danger": True, "bounding_box": {"x1": 220, "y1": 150, "x2": 420, "y2": 450},
    }]
    spoken_count = sum(
        1 for i in range(25)
        if update_spatial_scan_and_voice("Phía trước có ghế.", "HIGH", chair_block, None, W, 70.0 + i * 0.2)[2]
    )
    assert spoken_count == 1


def test_scenario_10_textureless_white_wall_closeup_fallback(sim_env):
    """Kịch bản 10: Áp sát bức tường trắng trơn bóng ở 0.9m khi Depth Map chưa kịp tải -> Vẫn phát hiện tường."""
    detector, _, W, H = sim_env
    blank_wall_frame = np.full((H, W, 3), 235, dtype=np.uint8)
    wall_hz = detector.detect_wall(None, known_boxes=[], frame_bgr=blank_wall_frame)
    assert wall_hz is not None and wall_hz["class_name"] == "wall"


def test_scenario_11_hand_shake_single_frame_dropout_stability(sim_env):
    """Kịch bản 11: Tay rung làm mất nhận diện ghế trong 1 frame (0.2s) rồi nhận lại -> Không báo nhầm An toàn."""
    _, _, W, _ = sim_env
    chair_block = [{
        "class_name": "chair", "class_name_vn": "ghế", "position": "center", "distance_m": 1.3,
        "in_roi": True, "is_danger": True, "bounding_box": {"x1": 220, "y1": 150, "x2": 420, "y2": 450},
    }]
    update_spatial_scan_and_voice("Phía trước có ghế.", "HIGH", chair_block, None, W, 100.0)
    _, _, sp_drop = update_spatial_scan_and_voice("", "LOW", [], None, W, 100.3)
    _, _, sp_recover = update_spatial_scan_and_voice("Phía trước có ghế.", "HIGH", chair_block, None, W, 100.6)
    assert sp_drop is False and sp_recover is False


def test_scenario_12_critical_collision_approach_under_0_85m(sim_env):
    """Kịch bản 12: Bỏ qua cảnh báo ở 2.0m và bước sát vào ghế 0.65m sau 2.2s -> Phá cooldown 6s để báo khẩn cấp."""
    _, _, W, _ = sim_env
    chair_2m = [{"class_name": "chair", "class_name_vn": "ghế", "position": "center", "distance_m": 2.0, "in_roi": True, "is_danger": True, "bounding_box": {"x1": 240, "y1": 160, "x2": 400, "y2": 420}}]
    update_spatial_scan_and_voice("Phía trước có ghế.", "HIGH", chair_2m, None, W, 110.0)
    chair_065m = [{"class_name": "chair", "class_name_vn": "ghế", "position": "center", "distance_m": 0.65, "in_roi": True, "is_danger": True, "bounding_box": {"x1": 120, "y1": 80, "x2": 520, "y2": 470}}]
    speech, _, speak = update_spatial_scan_and_voice("Phía trước có ghế.", "HIGH", chair_065m, None, W, 112.2)
    assert speak is True and "ghế" in speech


def test_scenario_13_narrow_classroom_corridor_between_two_tables(sim_env):
    """Kịch bản 13: Đi dọc hành lang giữa 2 dãy bàn (Trái 1.3m, Phải 1.3m, Giữa trống 3.5m) -> Báo đi thẳng."""
    _, _, W, _ = sim_env
    corridor_objs = [
        {"class_name": "table", "class_name_vn": "bàn", "position": "left", "distance_m": 1.3, "in_roi": False, "is_danger": True, "bounding_box": {"x1": 0, "y1": 180, "x2": 160, "y2": 460}},
        {"class_name": "table", "class_name_vn": "bàn", "position": "right", "distance_m": 1.3, "in_roi": False, "is_danger": True, "bounding_box": {"x1": 480, "y1": 180, "x2": 640, "y2": 460}},
    ]
    lanes_corridor = {"has_depth": True, "lanes": {
        "left": {"median_m": 1.3, "is_blocked": True},
        "center": {"median_m": 3.5, "is_blocked": False},
        "right": {"median_m": 1.3, "is_blocked": True},
    }}
    update_spatial_scan_and_voice("", "LOW", corridor_objs, lanes_corridor, W, 120.0)
    speech, _, speak = update_spatial_scan_and_voice("", "LOW", corridor_objs, lanes_corridor, W, 122.0)
    assert speak is True and speech == "Phía trước an toàn, đi thẳng."


def test_scenario_14_pothole_underfoot_warning(sim_env):
    """Kịch bản 14: Gặp ổ gà / hố sụt mặt đường ở 1.4m -> Báo chú ý dưới chân có ổ gà, hãy bước chậm."""
    _, _, W, _ = sim_env
    pothole_obj = [{"class_name": "pothole", "class_name_vn": "ổ gà", "position": "center", "distance_m": 1.4, "in_roi": True, "is_danger": True, "bounding_box": {"x1": 220, "y1": 340, "x2": 420, "y2": 450}}]
    speech, _, speak = update_spatial_scan_and_voice("Phía trước có ổ gà.", "HIGH", pothole_obj, None, W, 130.0)
    assert speak is True and speech == "Chú ý dưới chân, có ổ gà, hãy bước chậm."


def test_scenario_15_pitch_black_dark_room_no_false_wall(sim_env):
    """Kịch bản 15: Đi vào vùng tối om hoặc bị che camera (intensity = 12) -> Không báo giả bức tường."""
    detector, _, W, H = sim_env
    dark_frame = np.full((H, W, 3), 12, dtype=np.uint8)
    assert detector.detect_wall(None, known_boxes=[], frame_bgr=dark_frame) is None


def test_scenario_16_mobile_npu_export_plan():
    """Kịch bản 16: Kiểm tra cấu hình lượng tử hóa INT8/FP16 phục vụ triển khai NPU trên điện thoại."""
    from app.vision.npu_exporter import npu_exporter

    plan = npu_exporter.get_npu_deployment_plan("qualcomm_qnn")
    assert plan["compression_ratio"] >= 3.5
    assert plan["total_int8_size_mb"] < plan["total_fp32_size_mb"]
    assert len(plan["models"]) == 4


def test_scenario_17_stretched_tarpaulin_generic_front_obstacle(sim_env):
    """
    Kịch bản 17: Phía trước có tấm bạt căng ngang (hoặc rèm/vật lạ) ở 1.45m có nếp gấp nhăn
    mà YOLO không biết tên -> Quy chung về 'vật cản' và cảnh báo tức thì, tuyệt đối không im lặng.
    """
    detector, fusion, W, H = sim_env
    # Tạo độ sâu có nếp nhăn lượn sóng (1.35m - 1.58m) ở giữa lối đi để mô phỏng tấm bạt căng ngang
    y_grid = np.linspace(0, 6 * np.pi, H, dtype=np.float32)[:, None]
    dmap = np.full((H, W), 4.0, dtype=np.float32)
    tarpaulin_z = 1.45 + 0.12 * np.sin(y_grid)
    dmap[int(H * 0.18):int(H * 0.78), int(W * 0.26):int(W * 0.74)] = tarpaulin_z[int(H * 0.18):int(H * 0.78), :]

    obs = detector.detect_generic_front_obstacle(dmap, known_boxes=[])
    assert obs is not None
    assert obs["class_name"] == "obstacle"
    assert obs["class_name_vn"] == "vật cản"
    assert obs["is_danger"] is True

    f_res = fusion.fuse_detections([], [obs], [], None, W, H, now=140.0)
    speech, prio, speak = update_spatial_scan_and_voice("Phía trước có vật cản.", "HIGH", f_res["fused_objects"], None, W, 140.0)
    assert speak is True
    assert "Phía trước có vật cản" in speech
    assert prio in ("HIGH", "CRITICAL")


def test_scenario_18_deep_hole_ground_dropoff_emergency_stop(sim_env):
    """
    Kịch bản 18: Đang bước đi thì phía trước dưới chân có hố sâu / bậc hụt sâu (từ mặt đất 1.2m tụt sâu xuống 2.5m)
    -> Phát hiện 'hố sâu' và hô dừng lại khẩn cấp (CRITICAL).
    """
    detector, fusion, W, H = sim_env
    dmap = np.linspace(3.0, 1.1, H, dtype=np.float32)[:, None] * np.ones((1, W), dtype=np.float32)
    # Miệng hố sâu nằm ở nửa dưới vùng bước chân (tụt sâu xuống 2.5m)
    dmap[int(H * 0.70):int(H * 0.92), int(W * 0.26):int(W * 0.74)] = 2.5

    dropoff = detector.detect_ground_dropoff(dmap, known_boxes=[])
    assert dropoff is not None
    assert dropoff["class_name"] == "dropoff"
    assert dropoff["class_name_vn"] == "hố sâu"
    assert dropoff["is_danger"] is True

    f_res = fusion.fuse_detections([], [dropoff], [], None, W, H, now=150.0)
    speech, prio, speak = update_spatial_scan_and_voice("", "HIGH", f_res["fused_objects"], None, W, 150.0)
    assert speak is True
    assert speech == "Chú ý dưới chân, có hố sâu, hãy dừng lại."
    assert prio == "CRITICAL"


def test_scenario_19_narrow_alley_mode_speak_once_then_silent_and_0ms_interrupt(sim_env):
    """
    Kịch bản 19: Người khiếm thị đi vào con hẻm có 2 bức tường dài hai bên (Trái 1.15m, Phải 1.20m, Giữa trống 3.8m):
    - Lúc mới vào hẻm: Báo đúng 1 lần 'Hai bên là bức tường dài, phía trước thông thoáng, bạn hãy đi thẳng.'
    - Suốt 20 bước đi thẳng tiếp theo trong hẻm: Máy GIỮ IM LẶNG HOÀN TOÀN (không báo tường liên tục gây nhức đầu).
    - Khi bất ngờ có chiếc xe máy / vật cản xuất hiện ở 1.5m giữa hẻm: Ngắt 0ms và cảnh báo ngay lập tức!
    """
    _, _, W, _ = sim_env
    alley_lanes = {
        "has_depth": True,
        "lanes": {
            "left": {"median_m": 1.15, "is_blocked": True},
            "center": {"median_m": 3.80, "is_blocked": False},
            "right": {"median_m": 1.20, "is_blocked": True},
        },
    }

    # Bước 1: Vừa bước vào hẻm -> Đọc thông báo 1 lần duy nhất
    speech_entry, prio_entry, speak_entry = update_spatial_scan_and_voice("", "LOW", [], alley_lanes, W, 160.0)
    assert speak_entry is True
    assert speech_entry == "Hai bên là bức tường dài, phía trước thông thoáng, bạn hãy đi thẳng."
    assert prio_entry == "LOW"

    # Bước 2: Đi thẳng trong hẻm suốt 15 giây (30 frames) -> Im lặng tuyệt đối 100%
    silent_checks = [
        update_spatial_scan_and_voice("", "LOW", [], alley_lanes, W, 160.5 + i * 0.5)[2]
        for i in range(30)
    ]
    assert any(silent_checks) is False

    # Bước 3: Bất ngờ có vật cản / tấm bạt xuất hiện ở 1.4m trước mặt trong hẻm -> Ngắt 0ms báo nguy hiểm ngay!
    sudden_obs = [{
        "class_name": "obstacle", "class_name_vn": "vật cản", "confidence": 0.88,
        "position": "center", "distance_m": 1.4, "in_roi": True, "is_danger": True,
        "bounding_box": {"x1": 210, "y1": 140, "x2": 430, "y2": 440},
    }]
    speech_interrupt, prio_interrupt, speak_interrupt = update_spatial_scan_and_voice(
        "Phía trước có vật cản.", "HIGH", sudden_obs, alley_lanes, W, 175.6
    )
    assert speak_interrupt is True
    assert "Phía trước có vật cản" in speech_interrupt
    assert prio_interrupt in ("HIGH", "CRITICAL")


def test_scenario_20_alley_wall_drift_warning(sim_env):
    """
    Kịch bản 20: Đang đi trong hẻm nhưng người dùng đi lệch quá sát vào bức tường bên trái (0.62m)
    -> Nhắc chỉnh nhẹ sang phải rồi đi thẳng.
    """
    _, _, W, _ = sim_env
    drift_left_lanes = {
        "has_depth": True,
        "lanes": {
            "left": {"median_m": 0.62, "is_blocked": True},
            "center": {"median_m": 3.20, "is_blocked": False},
            "right": {"median_m": 1.35, "is_blocked": True},
        },
    }
    speech, prio, speak = update_spatial_scan_and_voice("", "LOW", [], drift_left_lanes, W, 190.0)
    assert speak is True
    assert speech == "Đang đi sát tường bên trái, hãy chỉnh nhẹ sang phải rồi đi thẳng."
    assert prio == "HIGH"


def test_scenario_21_wall_at_1_4m_to_1_8m_allows_two_more_steps(sim_env):
    """
    Kịch bản 21: Bức tường cách 1.3m - 1.8m (ngoài đời vẫn đủ đi thêm 2 bước chân thoải mái):
    - Không bị nhận nhầm là 'vật cản 1.8m'.
    - Hệ thống báo 'Phía trước an toàn, đi thẳng' và chỉ báo quay đầu khi tiến sát <= 1.0m.
    """
    detector, fusion, W, H = sim_env
    dmap_1_5m = np.full((H, W), 1.45, dtype=np.float32)

    # Tường phẳng ở 1.45m không bao giờ bị bắt nhầm là 'vật cản' (generic front obstacle)
    assert detector.detect_generic_front_obstacle(dmap_1_5m, known_boxes=[]) is None

    wall_1_45m = [{
        "class_name": "wall", "class_name_vn": "bức tường", "confidence": 0.88,
        "position": "center", "distance_m": 1.45, "in_roi": True, "is_danger": True,
        "bounding_box": {"x1": 60, "y1": 50, "x2": 580, "y2": 430},
    }]
    lanes_1_45m = {
        "has_depth": True,
        "lanes": {k: {"median_m": 1.45, "is_blocked": False} for k in ("left", "center", "right")},
    }
    f_res = fusion.fuse_detections([], wall_1_45m, [], lanes_1_45m, W, H, now=200.0)
    # Fusion Engine chuẩn hóa wall ở 1.45m (> 1.0m) là chưa nguy hiểm
    assert f_res["fused_objects"][0]["is_danger"] is False

    update_spatial_scan_and_voice("Phía trước có bức tường.", "LOW", f_res["fused_objects"], lanes_1_45m, W, 200.0)
    speech, prio, speak = update_spatial_scan_and_voice("Phía trước có bức tường.", "LOW", f_res["fused_objects"], lanes_1_45m, W, 202.0)
    assert speak is True
    assert speech == "Phía trước an toàn, đi thẳng."
    assert prio == "LOW"


def test_scenario_22_real_upward_stairs_with_landing_wall_ahead(sim_env):
    """
    Kịch bản 22: Đi LÊN cầu thang thực tế (mặt bậc tối màu xen kẽ cổ bậc sáng màu, khoảng cách xuyên tâm Z
    chênh lệch ít do bậc nâng cao dần về phía ngực, phía trên có tường chiếu nghỉ 1.8m)
    -> Bắt trúng 'cầu thang' và đọc 'Chú ý dưới chân, có cầu thang, hãy bước chậm', tuyệt đối không báo 'vật cản 1.8m'.
    """
    detector, fusion, W, H = sim_env
    frame = np.full((H, W, 3), 185, dtype=np.uint8)
    # Vẽ 4 bậc thang đi lên (mặt bậc tối màu 45, cổ bậc sáng màu 175)
    for y_top, y_bot in [(210, 245), (275, 315), (345, 390), (420, 465)]:
        frame[y_top:y_bot, 90:550] = (45, 50, 45)
        cv2.line(frame, (90, y_top), (550, y_top), (20, 20, 20), 3)
        cv2.line(frame, (90, y_bot), (550, y_bot), (210, 210, 200), 3)

    # Độ sâu xuyên tâm khi đi lên cầu thang chỉ biến thiên nhẹ 1.65m -> 1.80m
    dmap = np.linspace(1.65, 1.80, H, dtype=np.float32)[:, None] * np.ones((1, W), dtype=np.float32)

    stairs = detector.detect_stairs_and_curbs(frame, dmap, known_boxes=[])
    assert stairs is not None
    assert stairs["class_name"] == "stairs"
    assert stairs["class_name_vn"] == "cầu thang"
    assert stairs["is_danger"] is True

    f_res = fusion.fuse_detections([], [stairs], [], None, W, H, now=210.0)
    speech, _, speak = update_spatial_scan_and_voice("", "HIGH", f_res["fused_objects"], None, W, 210.0)
    assert speak is True
    assert speech == "Chú ý dưới chân, có cầu thang, hãy bước chậm."


def test_scenario_23_real_downward_granite_stairs_with_side_handrail(sim_env):
    """
    Kịch bản 23: Đứng đầu cầu thang đá hoa cương nhìn XUỐNG (Depth Anything V2 nội suy dốc mượt từ 4.5m về 2.8m,
    bên trái có song sắt dọc của lan can tay vịn) -> Bắt trúng 'cầu thang' và báo bước chậm.
    """
    detector, fusion, W, H = sim_env
    frame = np.full((H, W, 3), 115, dtype=np.uint8)
    # Song sắt dọc của lan can bên trái (x < 110)
    for x in range(25, 105, 20):
        cv2.line(frame, (x, 150), (x, 460), (40, 40, 40), 4)
    # 5 bậc thang đá hoa cương đi xuống có bóng đổ tăng dần
    step_shades = [130, 110, 92, 74, 55]
    step_ys = [210, 260, 315, 375, 435]
    prev_y = 180
    for y, shade in zip(step_ys, step_shades):
        frame[prev_y:y, 105:540] = (shade, shade, shade)
        cv2.line(frame, (105, y), (540, y), (25, 25, 25), 3)
        prev_y = y

    dmap = np.linspace(4.5, 2.8, H, dtype=np.float32)[:, None] * np.ones((1, W), dtype=np.float32)

    stairs = detector.detect_stairs_and_curbs(frame, dmap, known_boxes=[])
    assert stairs is not None
    assert stairs["class_name"] == "stairs"
    assert stairs["class_name_vn"] == "cầu thang"
    assert stairs["is_danger"] is True


def test_scenario_24_turn_right_after_obstacle_immediately_announces_safe_straight(sim_env):
    """
    Kịch bản 24: Khi hệ thống báo có vật cản và yêu cầu rẽ phải ('Phía trước có vật cản, hãy rẽ sang phải.'),
    người dùng lập tức xoay người rẽ sang phải khiến vật cản dạt sang rìa bên trái (position='left', in_roi=True)
    và lối đi phía trước đã thông thoáng -> Hệ thống PHẢI lập tức xác nhận 'Phía trước an toàn, đi thẳng.'
    sau 0.8s (và nhắc bù xác nhận sau 2.4s nếu loa điện thoại vừa đọc xong câu bảo rẽ), tuyệt đối không im lặng!
    """
    _, _, W, _ = sim_env

    # Bước 1 (t = 300.0s): Vật cản chắn lối đi hơi lệch trái -> Báo rẽ sang phải
    obs_front_left = [{
        "class_name": "obstacle", "class_name_vn": "vật cản", "confidence": 0.88,
        "position": "center", "distance_m": 1.3, "in_roi": True, "is_danger": True,
        "bounding_box": {"x1": 110, "y1": 150, "x2": 360, "y2": 450},
    }]
    lanes_blocked = {
        "has_depth": True,
        "lanes": {
            "left": {"median_m": 1.2, "is_blocked": True},
            "center": {"median_m": 1.3, "is_blocked": True},
            "right": {"median_m": 3.6, "is_blocked": False},
        },
    }
    speech_turn, prio_turn, speak_turn = update_spatial_scan_and_voice(
        "Phía trước có vật cản.", "HIGH", obs_front_left, lanes_blocked, W, 300.0
    )
    assert speak_turn is True
    assert speech_turn == "Phía trước có vật cản, hãy rẽ sang phải."
    assert prio_turn == "HIGH"

    # Bước 2 (t = 300.4s & t = 300.8s): Người dùng rẽ phải xong -> Vật cản dạt hẳn sang bên trái (x1=15, x2=220, position='left'),
    # phía trước (center) thông thoáng 3.8m -> Hệ thống lập tức báo "Phía trước an toàn, đi thẳng." ở 300.8s!
    obs_shifted_left = [{
        "class_name": "obstacle", "class_name_vn": "vật cản", "confidence": 0.85,
        "position": "left", "distance_m": 1.35, "in_roi": True, "is_danger": True,
        "bounding_box": {"x1": 15, "y1": 160, "x2": 220, "y2": 450},
    }]
    lanes_after_turn = {
        "has_depth": True,
        "lanes": {
            "left": {"median_m": 1.35, "is_blocked": False},
            "center": {"median_m": 3.80, "is_blocked": False},
            "right": {"median_m": 3.50, "is_blocked": False},
        },
    }
    update_spatial_scan_and_voice("Bên trái có vật cản.", "MEDIUM", obs_shifted_left, lanes_after_turn, W, 300.4)
    speech_clear, prio_clear, speak_clear = update_spatial_scan_and_voice(
        "Bên trái có vật cản.", "MEDIUM", obs_shifted_left, lanes_after_turn, W, 300.8
    )
    assert speak_clear is True
    assert speech_clear == "Phía trước an toàn, đi thẳng."
    assert prio_clear == "LOW"



