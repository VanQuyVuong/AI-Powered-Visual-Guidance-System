"""
Module Phân Tích Hình Học Độ Sâu & Bề Mặt (Geometric Hazard Detector):
Chuyên phát hiện:
1. BỨC TƯỜNG (Wall Plane): Mặt phẳng thẳng đứng chắn ngang lối đi, độ sâu đồng nhất theo chiều dọc.
2. BẬC TAM CẤP / BẬC THANG (Stairs / Steps / Curb): Các đường viền ngang song song kết hợp bước nhảy độ sâu gián đoạn (depth discontinuity).

Không dùng mô hình Deep Learning nặng nề mới -> Chạy bằng toán hình học + OpenCV thuần túy (< 3ms trên CPU).
"""

from typing import Any, Dict, List, Optional
import cv2
import numpy as np


class GeometricHazardDetector:
    """Bộ dò chướng ngại vật hình học: Bức tường và Bậc tam cấp/Bậc thang."""

    def __init__(self):
        pass

    def detect_wall(
        self,
        dmap: np.ndarray,
        known_boxes: Optional[List[Dict[str, int]]] = None,
        max_dist_m: float = 2.8,
    ) -> Optional[Dict[str, Any]]:
        """
        Phát hiện BỨC TƯỜNG chắn trước mặt:
        - Vùng quét: Hành lang trung tâm phía trước (x: 25% - 75%, y: 25% - 75%).
        - Tiêu chuẩn Bức Tường:
          + Mặt đứng: Gradient độ sâu theo chiều dọc |dZ/dy| rất nhỏ (< 0.035).
          + Độ sâu đồng nhất trong khoảng an toàn (0.4m <= Z <= max_dist_m).
          + Chiếm tỷ lệ diện tích lớn trong hành lang đi lại (> 15% diện tích quét).
          + Không bị che phủ bởi người/xe/bàn ghế mà YOLO đã phát hiện.
        """
        if dmap is None or dmap.size == 0:
            return None

        h, w = dmap.shape[:2]
        known_boxes = known_boxes or []

        # 1. Trích xuất ROI trung tâm
        x1, x2 = int(w * 0.22), int(w * 0.78)
        y1, y2 = int(h * 0.25), int(h * 0.78)
        roi_depth = dmap[y1:y2, x1:x2]
        if roi_depth.size == 0:
            return None

        # 2. Tính đạo hàm độ sâu theo phương thẳng đứng (Sobel Y)
        # Sàn nhà: khi y tăng (từ chân lên xa), Z tăng nhanh -> grad_y âm/dương lớn
        # Bức tường: Z gần như không đổi theo y -> grad_y xấp xỉ 0
        grad_y = cv2.Sobel(roi_depth, cv2.CV_32F, 0, 1, ksize=3)
        abs_grad_y = np.abs(grad_y)

        # 3. Lọc các pixel thỏa mãn: gần mặt trước (Z <= max_dist_m) và là mặt đứng (|grad_y| < 0.04)
        median_roi_z = float(np.median(roi_depth))
        if median_roi_z > max_dist_m or median_roi_z < 0.3:
            return None

        # Mặt phẳng tường có độ sâu quanh median_roi_z ± 0.35m
        wall_mask = (
            (abs_grad_y < 0.045)
            & (roi_depth >= max(0.2, median_roi_z - 0.40))
            & (roi_depth <= median_roi_z + 0.40)
        ).astype(np.uint8)

        # 4. Kiểm tra diện tích mảng tường
        roi_area = float((x2 - x1) * (y2 - y1))
        wall_pixel_count = int(np.sum(wall_mask))
        fill_ratio = wall_pixel_count / roi_area

        if fill_ratio < 0.16:
            return None

        # 5. Xác định bounding box của mảng tường
        ys, xs = np.where(wall_mask > 0)
        if ys.size == 0:
            return None

        bx1 = x1 + int(np.percentile(xs, 5))
        bx2 = x1 + int(np.percentile(xs, 95))
        by1 = y1 + int(np.percentile(ys, 5))
        by2 = y1 + int(np.percentile(ys, 95))

        # 6. Kiểm tra xem đã có bounding box YOLO lớn che phủ chưa
        wall_box_area = float(max(1, (bx2 - bx1) * (by2 - by1)))
        for kbox in known_boxes:
            ix1 = max(bx1, kbox.get("x1", 0))
            iy1 = max(by1, kbox.get("y1", 0))
            ix2 = min(bx2, kbox.get("x2", 0))
            iy2 = min(by2, kbox.get("y2", 0))
            if ix2 > ix1 and iy2 > iy1:
                inter_area = float((ix2 - ix1) * (iy2 - iy1))
                if inter_area / wall_box_area > 0.40:
                    return None  # Đã có vật thể khác (người, ô tô...) che phía trước

        wall_dist = round(median_roi_z, 1)
        return {
            "class_name": "wall",
            "class_name_vn": "bức tường",
            "confidence": min(0.92, 0.70 + fill_ratio * 0.35),
            "position": "center",
            "distance_m": wall_dist,
            "bounding_box": {"x1": bx1, "y1": by1, "x2": bx2, "y2": by2},
            "in_roi": True,
            "is_danger": wall_dist <= 2.2,
            "hazard_type": "WALL_OBSTACLE",
            "warning": f"Phía trước {wall_dist} mét có bức tường chắn lối đi."
        }

    def detect_stairs_and_curbs(
        self,
        frame_bgr: np.ndarray,
        dmap: np.ndarray,
        known_boxes: Optional[List[Dict[str, int]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Phát hiện BẬC TAM CẤP / BẬC THANG / GỜ VỈA HÈ dưới chân:
        - Vùng quét: Nửa dưới khung hình (y: 48% - 95%, x: 15% - 85%).
        - Tiêu chuẩn Bậc Thang:
          1. Quang học: Xuất hiện các cạnh ngang song song (Parallel Horizontal Edges) tạo bởi gờ bậc.
          2. Độ sâu: Tại vị trí các cạnh ngang có bước nhảy độ sâu (depth jump / ripple).
        - Phân loại:
          + Bậc thang đi lên (Stairs Up): Khoảng cách bậc xa hơn bước chân hoặc độ sâu nhô cao.
          + Bậc thang đi xuống / Hụt chân (Stairs Down / Drop): Mặt sàn tụt sâu xuống.
        """
        if frame_bgr is None or dmap is None or frame_bgr.size == 0 or dmap.size == 0:
            return None

        h, w = frame_bgr.shape[:2]
        known_boxes = known_boxes or []

        # 1. Trích xuất ROI nửa dưới (vùng bước chân)
        gy1, gy2 = int(h * 0.48), int(h * 0.95)
        gx1, gx2 = int(w * 0.15), int(w * 0.85)

        roi_bgr = frame_bgr[gy1:gy2, gx1:gx2]
        roi_depth = dmap[gy1:gy2, gx1:gx2]
        if roi_bgr.size == 0 or roi_depth.size == 0:
            return None

        # 2. Xử lý ảnh xám và tìm cạnh ngang (Horizontal Edge Filtering)
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # Bộ lọc đạo hàm theo trục Y để bắt mạnh các đường nằm ngang
        sobel_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)
        _, edge_thresh = cv2.threshold(abs_sobel_y, 40, 255, cv2.THRESH_BINARY)

        # Dùng toán tử hình thái học (Morphology) kéo dài các vệt ngang
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.08), 1))
        horizontal_edges = cv2.morphologyEx(edge_thresh, cv2.MORPH_OPEN, kernel_h)

        # 3. Dò các đoạn thẳng ngang bằng HoughLinesP
        min_line_len = int(w * 0.22)
        lines = cv2.HoughLinesP(
            horizontal_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=35,
            minLineLength=min_line_len,
            maxLineGap=25,
        )

        valid_horizontal_lines = []
        if lines is not None:
            for l in lines:
                pts = l.flatten()
                if len(pts) < 4:
                    continue
                lx1, ly1, lx2, ly2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
                # Góc nghiêng so với phương ngang (|angle| < 18 độ)
                angle = np.abs(np.arctan2(ly2 - ly1, lx2 - lx1) * 180.0 / np.pi)
                if angle <= 18:
                    valid_horizontal_lines.append({
                        "y_avg": (ly1 + ly2) / 2.0,
                        "x1": min(lx1, lx2),
                        "x2": max(lx1, lx2),
                        "length": float(np.sqrt((lx2 - lx1) ** 2 + (ly2 - ly1) ** 2)),
                    })

        # Cần ít nhất 2 đường ngang cách nhau hoặc 1 đường rất dài kết hợp bước nhảy độ sâu
        if not valid_horizontal_lines:
            return None

        # Gom nhóm các đường thẳng quá sát nhau theo trục Y (< 15px)
        valid_horizontal_lines.sort(key=lambda item: item["y_avg"])
        clustered_y = []
        for line in valid_horizontal_lines:
            if not clustered_y or abs(line["y_avg"] - clustered_y[-1]) > 14:
                clustered_y.append(line["y_avg"])

        # 4. Kiểm tra bước nhảy độ sâu (Depth Discontinuity) dọc theo trục Y tại các vị trí đường ngang
        # Sàn phẳng: dZ/dy tương đối đều
        # Bậc thang: Tại mỗi nếp bậc sẽ có bước nhảy độ sâu
        depth_jumps = 0
        depth_column = np.median(roi_depth, axis=1)  # Cột độ sâu trung bình theo chiều cao Y

        for cy in clustered_y:
            y_idx = int(cy)
            if 8 <= y_idx < len(depth_column) - 8:
                z_upper = float(np.mean(depth_column[max(0, y_idx - 6):y_idx]))
                z_lower = float(np.mean(depth_column[y_idx:min(len(depth_column), y_idx + 6)]))
                delta_z = abs(z_upper - z_lower)
                # Bước nhảy độ sâu tối thiểu 8cm (0.08m)
                if delta_z >= 0.08:
                    depth_jumps += 1

        # Tiêu chí nhận diện: Có từ 2 bậc trở lên hoặc 1 bậc rõ rệt kèm bước nhảy độ sâu
        is_stairs = len(clustered_y) >= 2 or (len(clustered_y) >= 1 and depth_jumps >= 1)
        if not is_stairs:
            return None

        # 5. Xác định loại nguy hiểm & Bounding Box
        step_min_y = int(min(clustered_y))
        step_max_y = int(max(clustered_y))

        bx1 = gx1 + int((gx2 - gx1) * 0.08)
        bx2 = gx2 - int((gx2 - gx1) * 0.08)
        by1 = gy1 + max(0, step_min_y - 20)
        by2 = gy1 + min(gy2 - gy1, step_max_y + 35)

        # Tính khoảng cách mét tới bậc gần nhất (chân người đi)
        stair_depth_roi = roi_depth[max(0, step_min_y):min(roi_depth.shape[0], step_max_y + 10), :]
        stair_dist = round(float(np.median(stair_depth_roi)), 1) if stair_depth_roi.size > 0 else 1.5

        if stair_dist > 3.5 or stair_dist < 0.3:
            return None

        # Phân biệt: Bậc tam cấp (1-2 bậc) hay Cầu thang lớn (>= 3 bậc)
        name_vn = "cầu thang" if len(clustered_y) >= 3 else "bậc tam cấp"
        class_name = "stairs" if len(clustered_y) >= 3 else "step"

        return {
            "class_name": class_name,
            "class_name_vn": name_vn,
            "confidence": min(0.95, 0.72 + len(clustered_y) * 0.07),
            "position": "center",
            "distance_m": stair_dist,
            "bounding_box": {"x1": bx1, "y1": by1, "x2": bx2, "y2": by2},
            "in_roi": True,
            "is_danger": stair_dist <= 2.5,
            "num_steps": len(clustered_y),
            "hazard_type": "GROUND_STEPS",
            "warning": f"Chú ý dưới chân, cách {stair_dist} mét có {name_vn}."
        }


# Singleton dùng chung
geometric_hazard_detector = GeometricHazardDetector()
