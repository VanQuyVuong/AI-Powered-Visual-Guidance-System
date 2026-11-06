"""
Unit Tests cho Decision Engine.
Theo CODING_RULES: test logic độc lập với YOLO (dùng dữ liệu giả).
Theo TEST_PLAN.md: test Position, Guidance, và ROI.
"""
import pytest
import sys
import os

# Thêm đường dẫn backend vào sys.path để import được module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.vision.decision_engine import (
    determine_position,
    generate_guidance,
    build_navigation_speech,
    is_in_roi,
    estimate_distance,
)
from app.schemas.vision import Detection, BoundingBox


# ==========================================
# TEST 1: determine_position()
# Chia khung hình làm 3 phần: Left | Center | Right
# ==========================================

class TestDeterminePosition:
    """Test vị trí tương đối của vật thể trong ảnh (Left/Center/Right)."""

    def test_object_clearly_left(self):
        """Vật thể nằm rõ ràng bên trái (tâm bbox ở 1/6 ảnh)."""
        # bbox từ x=0 đến x=200 trên ảnh rộng 640 → tâm = 100 < 213 → LEFT
        assert determine_position(0, 200, 640) == "left"

    def test_object_clearly_right(self):
        """Vật thể nằm rõ ràng bên phải (tâm bbox ở 5/6 ảnh)."""
        # bbox từ x=500 đến x=640 → tâm = 570 > 427 → RIGHT
        assert determine_position(500, 640, 640) == "right"

    def test_object_clearly_center(self):
        """Vật thể nằm rõ ràng ở giữa."""
        # bbox từ x=250 đến x=400 → tâm = 325 → CENTER
        assert determine_position(250, 400, 640) == "center"

    def test_boundary_left_center(self):
        """Ranh giới giữa Left và Center (tâm đúng bằng 1/3 ảnh)."""
        # Ảnh rộng 900 → 1/3 = 300 → tâm = 300 → KHÔNG < 300 → CENTER
        assert determine_position(200, 400, 900) == "center"

    def test_boundary_center_right(self):
        """Ranh giới giữa Center và Right (tâm đúng bằng 2/3 ảnh)."""
        # Ảnh rộng 900 → 2/3 = 600 → tâm = 600 → KHÔNG > 600 → CENTER
        assert determine_position(500, 700, 900) == "center"

    def test_just_inside_left(self):
        """Vật thể vừa lọt vào vùng trái (tâm nhỏ hơn 1/3 một chút)."""
        # Ảnh rộng 900 → 1/3 = 300 → tâm = 299 → LEFT
        assert determine_position(249, 349, 900) == "left"

    def test_just_inside_right(self):
        """Vật thể vừa lọt vào vùng phải (tâm lớn hơn 2/3 một chút)."""
        # Ảnh rộng 900 → 2/3 = 600 → tâm = 601 → RIGHT
        assert determine_position(551, 651, 900) == "right"

    def test_full_width_object(self):
        """Vật thể chiếm toàn bộ chiều ngang → tâm ở giữa → CENTER."""
        assert determine_position(0, 640, 640) == "center"

    def test_different_image_widths(self):
        """Test với ảnh có chiều rộng khác nhau (1920px)."""
        # 1920 / 3 = 640 → tâm < 640 → LEFT
        assert determine_position(0, 500, 1920) == "left"
        # tâm = 960 → CENTER
        assert determine_position(800, 1120, 1920) == "center"
        # tâm = 1500 > 1280 → RIGHT
        assert determine_position(1400, 1600, 1920) == "right"


# ==========================================
# TEST 2: generate_guidance()
# Sinh câu hướng dẫn tiếng Việt từ danh sách Detection
# ==========================================

class TestGenerateGuidance:
    """Test việc sinh câu hướng dẫn tiếng Việt."""

    def _make_detection(self, class_name: str, confidence: float, position: str) -> Detection:
        """Helper: tạo Detection giả để test (không cần YOLO)."""
        return Detection(
            class_name=class_name,
            confidence=confidence,
            position=position,
            bounding_box=BoundingBox(x1=0, y1=0, x2=100, y2=100)
        )

    def test_person_center(self):
        """person + center → 'Phía trước có người.'"""
        dets = [self._make_detection("person", 0.91, "center")]
        result = generate_guidance(dets)
        assert result is not None
        assert "người" in result["text"]
        assert "phía trước" in result["text"].lower()

    def test_car_left(self):
        """car + left → guidance có 'bên trái'."""
        dets = [self._make_detection("car", 0.88, "left")]
        result = generate_guidance(dets)
        assert result is not None
        assert "bên trái" in result["text"].lower()
        assert "xe ô tô" in result["text"].lower() or "xe" in result["text"].lower()

    def test_motorcycle_right(self):
        """motorcycle + right → guidance có 'bên phải'."""
        dets = [self._make_detection("motorcycle", 0.85, "right")]
        result = generate_guidance(dets)
        assert result is not None
        assert "bên phải" in result["text"].lower()

    def test_empty_detections(self):
        """Không có vật thể → trả None, không sinh câu rỗng."""
        result = generate_guidance([])
        assert result is None

    def test_low_confidence_filtered(self):
        """Vật thể confidence < 0.5 → bị lọc, không đọc."""
        dets = [self._make_detection("person", 0.3, "center")]
        result = generate_guidance(dets)
        assert result is None

    def test_multiple_objects(self):
        """Nhiều vật thể → guidance chứa 'và'."""
        dets = [
            self._make_detection("person", 0.91, "center"),
            self._make_detection("car", 0.88, "left"),
        ]
        result = generate_guidance(dets)
        assert result is not None
        assert "và" in result["text"]

    def test_center_has_high_priority(self):
        """Vật thể ở center (phía trước) → priority HIGH."""
        dets = [self._make_detection("person", 0.91, "center")]
        result = generate_guidance(dets)
        assert result["priority"] == "HIGH"

    def test_side_has_medium_priority(self):
        """Vật thể ở left/right → priority MEDIUM."""
        dets = [self._make_detection("car", 0.88, "left")]
        result = generate_guidance(dets)
        assert result["priority"] == "MEDIUM"

    def test_guidance_ends_with_period(self):
        """Câu hướng dẫn luôn kết thúc bằng dấu chấm."""
        dets = [self._make_detection("person", 0.91, "center")]
        result = generate_guidance(dets)
        assert result["text"].endswith(".")

    def test_unknown_class_uses_english_name(self):
        """Class chưa có trong vocab → dùng tên tiếng Anh gốc."""
        dets = [self._make_detection("umbrella", 0.80, "center")]
        result = generate_guidance(dets)
        assert result is not None
        assert "umbrella" in result["text"]

    def test_door_blocked_by_obstacle(self):
        """Vật cản (ghế) chắn trước cửa ra vào → sinh câu cảnh báo chắn ngang cửa."""
        chair = Detection(
            class_name="chair",
            confidence=0.85,
            position="center",
            distance_m=1.2,
            bounding_box=BoundingBox(x1=200, y1=200, x2=400, y2=400)
        )
        door = Detection(
            class_name="door",
            confidence=0.88,
            position="center",
            distance_m=2.5,
            bounding_box=BoundingBox(x1=150, y1=100, x2=450, y2=450)
        )
        result = generate_guidance([chair, door])
        assert result is not None
        assert "chắn ngang cửa ra vào" in result["text"].lower()
        assert "ghế" in result["text"].lower()
        assert result["priority"] == "HIGH"


# ==========================================
# TEST 3: is_in_roi()
# Lọc vật thể nằm ngoài Vùng Quan Tâm (ROI)
# ==========================================

class TestIsInROI:
    """Test thuật toán Vùng Quan Tâm (Region of Interest)."""

    def test_object_in_center_bottom(self):
        """Vật thể ở giữa, nửa dưới → NẰM TRONG ROI."""
        # Ảnh 640x480. Vật thể: tâm X=320 (giữa), Y dưới=400 (nửa dưới)
        assert is_in_roi(250, 200, 390, 400, 640, 480) is True

    def test_object_floating_top(self):
        """Vật thể lơ lửng nửa trên (y2 < 40% ảnh) → NGOÀI ROI."""
        # Ảnh 640x480. 40% chiều cao = 192. y2=150 < 192 → Bỏ qua
        assert is_in_roi(250, 50, 390, 150, 640, 480) is False

    def test_object_far_left_edge(self):
        """Vật thể tít bên lề trái (tâm X < 20% ảnh) → NGOÀI ROI."""
        # Ảnh 640x480. 20% chiều rộng = 128. Tâm X = (0+100)/2 = 50 < 128 → Bỏ qua
        assert is_in_roi(0, 200, 100, 400, 640, 480) is False

    def test_object_far_right_edge(self):
        """Vật thể tít bên lề phải (tâm X > 80% ảnh) → NGOÀI ROI."""
        # Ảnh 640x480. 80% chiều rộng = 512. Tâm X = (550+640)/2 = 595 > 512 → Bỏ qua
        assert is_in_roi(550, 200, 640, 400, 640, 480) is False

    def test_object_just_inside_roi_boundary(self):
        """Vật thể vừa lọt vào ROI (tâm X = 21%, y2 = 41% ảnh)."""
        # Ảnh 1000x1000. 20% = 200, 40% = 400
        # Tâm X = (200+400)/2 = 300 > 200 ✓, y2 = 410 > 400 ✓ → Trong ROI
        assert is_in_roi(200, 300, 400, 410, 1000, 1000) is True

    def test_sky_object_ignored(self):
        """Vật thể trên trời (bird, airplane) thường y2 rất nhỏ → NGOÀI ROI."""
        # Chim bay: y dưới cùng = 100 trên ảnh 640x480 → 100 < 192 → Bỏ qua
        assert is_in_roi(300, 50, 350, 100, 640, 480) is False


# ==========================================
# TEST 4: estimate_distance()
# Ước lượng khoảng cách từ tỷ lệ bbox
# ==========================================

class TestEstimateDistance:
    """Test ước lượng khoảng cách dựa trên tỷ lệ bounding box."""

    def test_very_close(self):
        """Vật thể chiếm > 60% chiều cao ảnh → 'rất gần'."""
        # Ảnh 480px cao. Vật thể y1=50, y2=400 → chiều cao = 350 → 350/480 = 0.73 > 0.6
        assert estimate_distance(50, 400, 480) == "rất gần"

    def test_medium_distance(self):
        """Vật thể chiếm 30-60% chiều cao ảnh → 'ở khoảng cách vừa'."""
        # Vật thể y1=150, y2=350 → chiều cao = 200 → 200/480 = 0.42
        assert estimate_distance(150, 350, 480) == "ở khoảng cách vừa"

    def test_far_away(self):
        """Vật thể chiếm < 30% chiều cao ảnh → 'ở xa'."""
        # Vật thể y1=200, y2=300 → chiều cao = 100 → 100/480 = 0.21
        assert estimate_distance(200, 300, 480) == "ở xa"


# ==========================================
# TEST 5: Gom nhóm vật cản & Dẫn đường thông minh
# ==========================================

class TestClusteringAndNavigation:
    """Test gom nhóm nhiều vật cản gần nhau và dẫn đường né vật cản."""

    def test_multiple_bottles_grouped_into_nhieu_chai_nuoc(self):
        """4 chai nước đặt gần nhau phía trước → gom gọn thành 1 cụm 'nhiều chai nước' (không đọc lặp từng chai)."""
        bottles = [
            Detection(class_name="bottle", confidence=0.82, position="center", distance_m=1.2,
                      bounding_box=BoundingBox(x1=220, y1=180, x2=250, y2=280)),
            Detection(class_name="bottle", confidence=0.79, position="center", distance_m=1.3,
                      bounding_box=BoundingBox(x1=270, y1=185, x2=300, y2=285)),
            Detection(class_name="bottle", confidence=0.75, position="center", distance_m=1.3,
                      bounding_box=BoundingBox(x1=320, y1=180, x2=350, y2=280)),
            Detection(class_name="bottle", confidence=0.71, position="right", distance_m=1.4,
                      bounding_box=BoundingBox(x1=440, y1=190, x2=470, y2=290)),
        ]
        res = generate_guidance(bottles)
        assert res is not None
        assert "nhiều chai nước" in res["text"].lower()
        assert res["text"].lower().count("chai nước") == 1

    def test_bottles_on_table_grouped_into_nhieu_chai_nuoc_tren_ban(self):
        """Nhiều chai nước trên bàn → báo 'Phía trước ... có nhiều chai nước trên bàn.'"""
        dets = [
            Detection(class_name="dining table", confidence=0.88, position="center", distance_m=1.3,
                      bounding_box=BoundingBox(x1=150, y1=220, x2=520, y2=450)),
            Detection(class_name="bottle", confidence=0.82, position="center", distance_m=1.2,
                      bounding_box=BoundingBox(x1=220, y1=180, x2=250, y2=280)),
            Detection(class_name="bottle", confidence=0.79, position="center", distance_m=1.3,
                      bounding_box=BoundingBox(x1=270, y1=185, x2=300, y2=285)),
        ]
        res = generate_guidance(dets)
        assert res is not None
        assert "nhiều chai nước trên bàn" in res["text"].lower()

    def test_mixed_objects_grouped_into_nhieu_vat_can(self):
        """Nhiều đồ vật khác loại đứng gần nhau → gom thành 'nhiều vật cản'."""
        dets = [
            Detection(class_name="chair", confidence=0.82, position="center", distance_m=1.2,
                      bounding_box=BoundingBox(x1=200, y1=180, x2=280, y2=380)),
            Detection(class_name="backpack", confidence=0.79, position="center", distance_m=1.3,
                      bounding_box=BoundingBox(x1=290, y1=220, x2=350, y2=380)),
            Detection(class_name="suitcase", confidence=0.75, position="center", distance_m=1.4,
                      bounding_box=BoundingBox(x1=360, y1=210, x2=430, y2=390)),
        ]
        res = generate_guidance(dets)
        assert res is not None
        assert "nhiều vật cản" in res["text"].lower()

    def test_safe_ahead_reports_di_thang(self):
        """Khi phía trước không có vật cản → báo 'Phía trước an toàn, đi thẳng.'"""
        res = build_navigation_speech(None, [], {"best_direction": "center", "Corridors": {}}, 640)
        assert "an toàn" in res["text"].lower()
        assert "đi thẳng" in res["text"].lower()

    def test_blocked_center_guides_to_clear_side(self):
        """Khi phía trước có bàn/vật cản và bên trái trống → hướng dẫn rẽ trái."""
        base = {"text": "Cảnh báo! Phía trước có bàn cách 1.2 mét.", "priority": "HIGH"}
        objs = [
            {"class_name": "dining table", "class_name_vn": "bàn", "position": "center",
             "distance_m": 1.2, "bbox": [220, 200, 520, 440]}
        ]
        nav = {"best_direction": "left", "Corridors": {"left": 3.5, "center": 1.2, "right": 1.4}}
        res = build_navigation_speech(base, objs, nav, 640)
        assert "rẽ trái" in res["text"].lower()

    def test_wall_blocking_all_lanes_tells_user_to_look_left_or_right(self):
        """Khi bức tường chắn ngang toàn bộ phía trước → nhắc dừng lại, nhìn sang trái hoặc sang phải."""
        base = {"text": "Cảnh báo! Phía trước có bức tường cách 1.1 mét.", "priority": "CRITICAL"}
        objs = [
            {"class_name": "wall", "class_name_vn": "bức tường", "position": "center",
             "distance_m": 1.1, "bbox": [60, 40, 580, 420]}
        ]
        nav = {"best_direction": "stop", "Corridors": {"left": 1.1, "center": 1.1, "right": 1.1}}
        res = build_navigation_speech(base, objs, nav, 640)
        assert "nhìn sang trái hoặc sang phải" in res["text"].lower()


# ==========================================
# Chạy tests: pytest backend/tests/test_decision_engine.py -v
# ==========================================

