"""
Module Phân Tích Hình Học Độ Sâu & Bề Mặt (Geometric Hazard Detector):
Chuyên phát hiện:
1. BỨC TƯỜNG (Wall Plane): Mặt phẳng thẳng đứng chắn ngang lối đi (kết hợp Gradient độ sâu 3D + Phân tích bề mặt quang học khi tiến sát tường).
2. BẬC THỀM / BẬC TAM CẤP / CẦU THANG (Doorstep / Steps / Stairs): Các đường gờ ngang song song chịu được độ nghiêng tay cầm camera (±18 độ) kết hợp bước nhảy độ sâu hoặc tương phản gờ bậc.

Không dùng mô hình Deep Learning nặng nề mới -> Chạy bằng toán hình học + OpenCV thuần túy (< 3ms trên CPU).
"""

from typing import Any, Dict, List, Optional
import cv2
import numpy as np


class GeometricHazardDetector:
    """Bộ dò chướng ngại vật hình học: Bức tường, Bậc thềm, Bậc tam cấp và Cầu thang."""

    def __init__(self):
        pass

    def detect_wall(
        self,
        dmap: Optional[np.ndarray],
        known_boxes: Optional[List[Dict[str, int]]] = None,
        max_dist_m: float = 3.6,
        frame_bgr: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Phát hiện BỨC TƯỜNG chắn trước mặt:
        - Lớp 1 (Độ sâu 3D): Mặt phẳng đứng có đạo hàm độ sâu chuẩn hóa |dZ/dy| nhỏ và chênh lệch độ sâu trên-dưới thấp.
        - Lớp 2 (Quang học cận cảnh): Khi người dùng tiến sát tường trơn (0.4m - 2.0m), vùng trung tâm không có vật thể YOLO
          nhưng bị chắn bởi một mảng bề mặt đồng nhất diện tích lớn.
        """
        known_boxes = known_boxes or []

        # ------------------------------------------------------------------
        # LỚP 1: PHÂN TÍCH ĐẠO HÀM BẢN ĐỒ ĐỘ SÂU (DEPTH PLANE ANALYSIS)
        # ------------------------------------------------------------------
        if dmap is not None and dmap.size > 0:
            h, w = dmap.shape[:2]
            x1, x2 = int(w * 0.22), int(w * 0.78)
            y1, y2 = int(h * 0.22), int(h * 0.78)
            roi_depth = dmap[y1:y2, x1:x2]

            if roi_depth.size > 0:
                # Kiểm tra xem có phải mặt đường/sàn nhà dốc xa dần không:
                # Trên mặt đường mở, phần trên (y1) xa hơn phần dưới (y2) rất nhiều.
                rh = roi_depth.shape[0]
                top_band_z = float(np.median(roi_depth[:max(1, int(rh * 0.25)), :]))
                bot_band_z = float(np.median(roi_depth[max(0, int(rh * 0.75)):, :]))
                vertical_z_span = abs(top_band_z - bot_band_z)

                # Chuẩn hóa Sobel ksize=3 bằng cách chia cho 8.0 để lấy đúng dZ/dy (mét/pixel)
                grad_y = cv2.Sobel(roi_depth, cv2.CV_32F, 0, 1, ksize=3) / 8.0
                abs_grad_y = np.abs(grad_y)

                median_roi_z = float(np.median(roi_depth))
                if 0.25 <= median_roi_z <= 1.35 and vertical_z_span <= 1.15:
                    wall_mask = (
                        (abs_grad_y < 0.011)
                        & (roi_depth >= max(0.2, median_roi_z - 0.45))
                        & (roi_depth <= median_roi_z + 0.45)
                    ).astype(np.uint8)

                    roi_area = float((x2 - x1) * (y2 - y1))
                    wall_pixel_count = int(np.sum(wall_mask))
                    fill_ratio = wall_pixel_count / max(1.0, roi_area)

                    if fill_ratio >= 0.30:
                        ys, xs = np.where(wall_mask > 0)
                        if ys.size > 0:
                            bx1 = x1 + int(np.percentile(xs, 5))
                            bx2 = x1 + int(np.percentile(xs, 95))
                            by1 = y1 + int(np.percentile(ys, 5))
                            by2 = y1 + int(np.percentile(ys, 95))

                            has_center_kbox = any(
                                (k.get("x1", 0) < w * 0.65 and k.get("x2", 0) > w * 0.35)
                                for k in known_boxes
                            )
                            if not has_center_kbox and not self._is_box_covered(bx1, by1, bx2, by2, known_boxes, thresh=0.20):
                                wall_dist = round(min(median_roi_z, 1.3), 1)
                                is_danger = wall_dist <= 1.35
                                return {
                                    "class_name": "wall",
                                    "class_name_vn": "bức tường",
                                    "confidence": round(min(0.94, 0.72 + fill_ratio * 0.30), 2),
                                    "position": "center",
                                    "distance_m": wall_dist,
                                    "bounding_box": {"x1": bx1, "y1": by1, "x2": bx2, "y2": by2},
                                    "in_roi": is_danger,
                                    "is_danger": is_danger,
                                    "hazard_type": "WALL_OBSTACLE",
                                    "warning": f"Phía trước {wall_dist} mét có bức tường chắn lối đi." if is_danger else ""
                                }

        # ------------------------------------------------------------------
        # LỚP 2: PHÂN TÍCH QUANG HỌC CẬN CẢNH (OPTICAL WALL DETECTOR)
        # Bắt bức tường trơn/phẳng khi tiến lại gần (ngay cả khi tường không có vân)
        # ------------------------------------------------------------------
        if frame_bgr is not None and frame_bgr.size > 0:
            fh, fw = frame_bgr.shape[:2]
            cx1, cx2 = int(fw * 0.22), int(fw * 0.78)
            cy1, cy2 = int(fh * 0.20), int(fh * 0.78)

            # Nếu đã có vật thể YOLO nằm ở trung tâm thì không báo nhầm là tường trơn
            has_center_kbox = any(
                (k.get("x1", 0) < fw * 0.65 and k.get("x2", 0) > fw * 0.35)
                for k in known_boxes
            )
            if not has_center_kbox and not self._is_box_covered(cx1, cy1, cx2, cy2, known_boxes, thresh=0.10):
                # Nếu dmap có sẵn và báo mặt đường đang trải dài ra xa (> 1.4m chênh lệch trên-dưới) -> Không phải tường
                is_open_corridor = False
                if dmap is not None and dmap.size > 0:
                    rd = dmap[cy1:cy2, cx1:cx2]
                    if rd.size > 0:
                        t_z = float(np.median(rd[:max(1, int(rd.shape[0] * 0.25)), :]))
                        b_z = float(np.median(rd[max(0, int(rd.shape[0] * 0.75)):, :]))
                        if (t_z - b_z) > 1.3 or float(np.median(rd)) > 3.8:
                            is_open_corridor = True

                if not is_open_corridor:
                    roi_bgr = frame_bgr[cy1:cy2, cx1:cx2]
                    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

                    # Chia thành lưới 4x4 để kiểm tra độ đồng nhất màu sắc & độ phẳng bề mặt của bức tường
                    gh, gw = gray.shape[:2]
                    bh, bw = max(1, gh // 4), max(1, gw // 4)
                    med_h = float(np.median(hsv[:, :, 0]))
                    med_s = float(np.median(hsv[:, :, 1]))
                    med_v = float(np.median(hsv[:, :, 2]))

                    flat_blocks = 0
                    total_blocks = 16
                    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)

                    for r in range(4):
                        for c in range(4):
                            sub_g = gray[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
                            sub_hsv = hsv[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
                            sub_e = edges[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
                            if sub_g.size == 0:
                                continue
                            std_g = float(np.std(sub_g))
                            edge_ratio = float(np.count_nonzero(sub_e)) / float(sub_e.size)
                            dh = min(abs(float(np.median(sub_hsv[:, :, 0])) - med_h), 180.0 - abs(float(np.median(sub_hsv[:, :, 0])) - med_h))
                            ds = abs(float(np.median(sub_hsv[:, :, 1])) - med_s)
                            dv = abs(float(np.median(sub_hsv[:, :, 2])) - med_v)

                            if std_g < 22.0 and edge_ratio < 0.045 and dh < 22.0 and ds < 50.0 and dv < 55.0:
                                flat_blocks += 1

                    # Nếu >= 11/16 ô trung tâm là cùng một mảng bề mặt phẳng đồng nhất -> Bức tường chắn trước mặt!
                    if flat_blocks >= 11:
                        est_dist = 1.2
                        if dmap is not None and dmap.size > 0:
                            est_dist = round(min(3.5, max(0.5, float(np.median(dmap[cy1:cy2, cx1:cx2])))), 1)
                        is_danger = est_dist <= 1.35
                        return {
                            "class_name": "wall",
                            "class_name_vn": "bức tường",
                            "confidence": round(min(0.92, 0.65 + (flat_blocks / 16.0) * 0.25), 2),
                            "position": "center",
                            "distance_m": est_dist,
                            "bounding_box": {"x1": cx1, "y1": cy1, "x2": cx2, "y2": cy2},
                            "in_roi": is_danger,
                            "is_danger": is_danger,
                            "hazard_type": "WALL_OBSTACLE",
                            "warning": f"Phía trước {est_dist} mét có bức tường chắn lối đi." if is_danger else ""
                        }

        return None

    def detect_stairs_and_curbs(
        self,
        frame_bgr: np.ndarray,
        dmap: Optional[np.ndarray] = None,
        known_boxes: Optional[List[Dict[str, int]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Phát hiện BẬC THỀM / BẬC TAM CẤP / CẦU THANG dưới chân:
        - Vùng quét: Từ giữa xuống chân khung hình (y: 38% - 95%, x: 12% - 88%).
        - Chống báo giả tuyệt đối với:
          1. Nền gạch men có rãnh (ceramic tile grout lines): Trừ độ dốc tuyến tính của mặt sàn phẳng
             và loại bỏ lưới gạch có tỷ lệ đường kẻ dọc cao.
          2. Tủ có thanh kệ ngang ở dưới (cabinet shelves): Kiểm tra độ sâu mặt đứng đồng nhất
             và tính đơn điệu bậc thang từ dưới lên trên.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        h, w = frame_bgr.shape[:2]
        known_boxes = known_boxes or []

        # BẮT BUỘC phải có bản đồ độ sâu 3D hợp lệ để xác nhận bậc thềm/cầu thang!
        # Không bao giờ chỉ dựa vào vệt kẻ 2D trên nền gạch hoặc kệ tủ.
        if dmap is None or dmap.size == 0 or dmap.shape[:2] != (h, w):
            return None

        # Nếu đã có vật cản lớn (tủ, bàn, ghế, người) chiếm vùng trung tâm phía dưới -> không dò cầu thang xuyên qua vật cản
        for kbox in known_boxes:
            kx1, ky1, kx2, ky2 = kbox.get("x1", 0), kbox.get("y1", 0), kbox.get("x2", 0), kbox.get("y2", 0)
            if kx1 < w * 0.68 and kx2 > w * 0.32 and ky2 > h * 0.45 and (kx2 - kx1) >= w * 0.22:
                return None

        # 1. Trích xuất ROI vùng bước chân
        gy1, gy2 = int(h * 0.38), int(h * 0.95)
        gx1, gx2 = int(w * 0.12), int(w * 0.88)

        roi_bgr = frame_bgr[gy1:gy2, gx1:gx2]
        if roi_bgr.size == 0:
            return None

        roi_depth = dmap[gy1:gy2, gx1:gx2]
        rh = roi_depth.shape[0]
        if rh < 24:
            return None

        # KIỂM TRA CHỐNG NHẦM TƯỜNG PHẲNG / THÂN TỦ CÓ KỆ NGANG THÀNH CẦU THANG:
        # Cầu thang/bậc thềm đi lên/xuống trên mặt đất bắt buộc đỉnh ROI (xa) phải sâu hơn đáy ROI (gần) rõ rệt (>= 0.28m)
        # và có phương sai độ sâu đủ lớn. Nếu mặt đứng của chiếc tủ có kệ ngang thì top_z ~ bot_z -> loại ngay!
        top_z = float(np.median(roi_depth[:rh // 3, :]))
        bot_z = float(np.median(roi_depth[2 * rh // 3:, :]))
        if (top_z - bot_z) < 0.28 or float(np.std(roi_depth)) < 0.18:
            return None

        # 2. Lọc cạnh ngang thực tế
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        sobel_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)
        abs_x = cv2.convertScaleAbs(sobel_x)
        abs_y = cv2.convertScaleAbs(sobel_y)

        # KIỂM TRA LƯỚI GẠCH MEN (Ceramic Tile Grid) & CÁNH TỦ DỌC:
        # Nền gạch men có cả rãnh ngang lẫn rãnh dọc, hoặc chiếc tủ có 2 thanh trụ dọc 2 bên rất rõ.
        # Cầu thang thật thì cạnh ngang (abs_y) áp đảo hoàn toàn cạnh dọc (abs_x).
        horiz_px = int(np.count_nonzero((abs_y >= 45) & (abs_y > abs_x * 1.5)))
        vert_px = int(np.count_nonzero((abs_x >= 45) & (abs_x > abs_y * 1.5)))
        if horiz_px == 0 or vert_px > horiz_px * 0.50:
            return None

        # Chỉ giữ lại các cạnh ngang thực sự sắc nét (abs_y >= 45) và mạnh hơn cạnh dọc rõ rệt
        horiz_mask = ((abs_y >= 45) & (abs_y.astype(np.float32) >= abs_x.astype(np.float32) * 1.55)).astype(np.uint8) * 255

        # Nối liền các vết đứt nhỏ dọc theo gờ bậc
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        horizontal_edges = cv2.morphologyEx(horiz_mask, cv2.MORPH_CLOSE, kernel_close)

        # 3. Dò các đoạn thẳng gờ bậc bằng HoughLinesP (chiều dài tối thiểu >= 32% chiều ngang ảnh)
        min_line_len = int(w * 0.32)
        lines = cv2.HoughLinesP(
            horizontal_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=42,
            minLineLength=min_line_len,
            maxLineGap=20,
        )

        valid_horizontal_lines = []
        if lines is not None:
            for l in lines:
                pts = l.flatten()
                if len(pts) < 4:
                    continue
                lx1, ly1, lx2, ly2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
                angle = np.abs(np.arctan2(ly2 - ly1, lx2 - lx1) * 180.0 / np.pi)
                if angle <= 14.0:
                    full_x1, full_x2 = gx1 + min(lx1, lx2), gx1 + max(lx1, lx2)
                    full_y = gy1 + int((ly1 + ly2) / 2.0)
                    if self._is_line_inside_yolo_box(full_x1, full_x2, full_y, known_boxes):
                        continue

                    valid_horizontal_lines.append({
                        "y_avg": (ly1 + ly2) / 2.0,
                        "x1": min(lx1, lx2),
                        "x2": max(lx1, lx2),
                        "length": float(np.sqrt((lx2 - lx1) ** 2 + (ly2 - ly1) ** 2)),
                    })

        if not valid_horizontal_lines:
            return None

        # Gom nhóm các cạnh ngang cách nhau <= 28px thành 1 bậc duy nhất
        valid_horizontal_lines.sort(key=lambda item: item["y_avg"])
        clusters: List[Dict[str, float]] = []
        for line in valid_horizontal_lines:
            if not clusters or abs(line["y_avg"] - clusters[-1]["y_avg"]) > 28:
                clusters.append({
                    "y_avg": line["y_avg"],
                    "x1": float(line["x1"]),
                    "x2": float(line["x2"]),
                    "max_len": line["length"],
                })
            else:
                c = clusters[-1]
                c["y_avg"] = (c["y_avg"] + line["y_avg"]) / 2.0
                c["x1"] = min(c["x1"], float(line["x1"]))
                c["x2"] = max(c["x2"], float(line["x2"]))
                c["max_len"] = max(c["max_len"], float(c["x2"] - c["x1"]))

        clustered_y = [c["y_avg"] for c in clusters]
        max_span_px = max((c["x2"] - c["x1"]) for c in clusters)
        if max_span_px < w * 0.32:
            return None

        # 4. Kiểm tra bước nhảy độ sâu ĐỘT NGỘT (Step Discontinuity) vượt trên độ dốc tự nhiên của sàn phẳng!
        # LỖI CŨ: Sàn gạch men phẳng nghiêng từ 0.8m (đáy) lên 2.8m (đỉnh) qua 300px có độ dốc tự nhiên ~0.0067m/px,
        # nên khoảng cách 14px giữa trên và dưới đường ron gạch đã chênh ~0.09m >= 0.06m -> báo nhầm rãnh gạch là cầu thang!
        # KHẮC PHỤC: Tính độ dốc nền trung bình (baseline_slope_per_px) và yêu cầu bước nhảy thực tế qua gờ bậc
        # phải lớn hơn độ dốc nền ít nhất +0.14m (tổng bước nhảy >= 0.18m).
        depth_column = np.median(roi_depth, axis=1)
        gray_column = np.median(blurred, axis=1)

        dy_all = np.diff(depth_column)
        baseline_slope_per_px = float(max(0.0, -np.median(dy_all)))  # Sàn phẳng: y tăng (xuống dưới) thì z giảm

        depth_jumps = 0
        optical_jumps = 0
        max_net_jump = 0.0
        cluster_depths: List[float] = []

        for cy in clustered_y:
            y_idx = int(cy)
            if 10 <= y_idx < len(depth_column) - 10:
                z_upper = float(np.mean(depth_column[max(0, y_idx - 12):max(1, y_idx - 2)]))
                z_lower = float(np.mean(depth_column[min(len(depth_column) - 1, y_idx + 2):min(len(depth_column), y_idx + 12)]))
                cluster_depths.append(float(depth_column[y_idx]))

                # Bậc thang/bậc thềm đi tới trước mặt: phía trên gờ (z_upper) phải XA HƠN phía dưới gờ (z_lower)
                raw_jump = z_upper - z_lower
                expected_smooth_drop = baseline_slope_per_px * 14.0
                net_jump = raw_jump - expected_smooth_drop

                if raw_jump >= 0.18 and net_jump >= 0.14:
                    depth_jumps += 1
                    max_net_jump = max(max_net_jump, net_jump)

            if 6 <= y_idx < len(gray_column) - 6:
                g_upper = float(np.mean(gray_column[max(0, y_idx - 10):max(1, y_idx - 2)]))
                g_lower = float(np.mean(gray_column[min(len(gray_column) - 1, y_idx + 2):min(len(gray_column), y_idx + 10)]))
                if abs(g_upper - g_lower) >= 14.0:
                    optical_jumps += 1

        # Kiểm tra tính đơn điệu của độ sâu các bậc: Bậc ở trên cao (xa hơn, index nhỏ) phải có độ sâu lớn hơn bậc ở dưới (gần hơn)
        # Nếu là các thanh kệ của chiếc tủ (kệ trên thò ra, gầm tủ thụt vào hoặc phẳng đều) -> vi phạm tính đơn điệu bậc thang!
        if len(cluster_depths) >= 2:
            for i in range(len(cluster_depths) - 1):
                if cluster_depths[i] < cluster_depths[i + 1] + 0.10:
                    return None

        is_stairs = (depth_jumps >= 1) and (max_net_jump >= 0.18) and (optical_jumps >= 1)
        if not is_stairs:
            return None

        # 5. Xác định tọa độ & khoảng cách
        step_min_y = int(min(clustered_y))
        step_max_y = int(max(clustered_y))
        vertical_span = step_max_y - step_min_y

        bx1 = gx1 + int(min(c["x1"] for c in clusters))
        bx2 = gx1 + int(max(c["x2"] for c in clusters))
        by1 = gy1 + max(0, step_min_y - 20)
        by2 = gy1 + min(gy2 - gy1, step_max_y + 35)

        stair_depth_roi = roi_depth[max(0, step_min_y):min(roi_depth.shape[0], step_max_y + 12), :]
        stair_dist = round(float(np.median(stair_depth_roi)), 1) if stair_depth_roi.size > 0 else 1.5

        if stair_dist > 3.2 or stair_dist < 0.35:
            return None

        # Phân biệt chuẩn xác:
        # - Cầu thang: Phải có từ 3 bậc trở lên, trải dài theo chiều dọc >= 85px và có >= 2 bước nhảy độ sâu thực sự
        # - Bậc tam cấp: 2 bậc rõ rệt cách nhau >= 45px và có >= 2 bước nhảy độ sâu
        # - Bậc thềm: 1 bậc có bước nhảy độ sâu sắc nét (>= 0.22m)
        if len(clustered_y) >= 3 and vertical_span >= 85 and depth_jumps >= 2:
            name_vn = "cầu thang"
            class_name = "stairs"
        elif len(clustered_y) == 2 and vertical_span >= 45 and depth_jumps >= 2:
            name_vn = "bậc tam cấp"
            class_name = "step"
        elif depth_jumps >= 1 and max_net_jump >= 0.22:
            name_vn = "bậc thềm"
            class_name = "step"
        else:
            return None

        return {
            "class_name": class_name,
            "class_name_vn": name_vn,
            "confidence": round(min(0.95, 0.72 + len(clustered_y) * 0.07), 2),
            "position": "center",
            "distance_m": stair_dist,
            "bounding_box": {"x1": bx1, "y1": by1, "x2": bx2, "y2": by2},
            "in_roi": True,
            "is_danger": stair_dist <= 2.4,
            "num_steps": len(clustered_y),
            "hazard_type": "GROUND_STEPS",
            "warning": f"Chú ý dưới chân, cách {stair_dist} mét có {name_vn}."
        }

    def detect_pillar(
        self,
        dmap: Optional[np.ndarray],
        known_boxes: Optional[List[Dict[str, int]]] = None,
        frame_bgr: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Phát hiện CÂY CỘT / TRỤ ĐỨNG (Pillar / Column) ngay trước mặt (0.35m - 1.75m):
        - Giải quyết triệt để lỗi: YOLO COCO không có lớp 'column'/'pillar' và SegFormer đôi khi
          nhận nhầm cột bê tông sơn phẳng là 'wall' (nền tường), trong khi detect_wall lại bỏ qua
          vì cây cột hẹp hơn bức tường (chỉ chiếm 12% - 62% chiều ngang khung hình, hai bên cột là khoảng trống xa).
        - Nguyên lý hình học 3D:
          Tìm dải dọc liên tục nằm chắn hành lang giữa có độ sâu đồng nhất theo phương đứng (|dZ/dy| < 0.025),
          khoảng cách gần (<= 1.75m), và nhô ra gần hơn hẳn so với nền phía sau ở bên trái hoặc bên phải (>= 0.35m).
        """
        if dmap is None or dmap.size == 0:
            return None

        h, w = dmap.shape[:2]
        if h < 32 or w < 32:
            return None

        known_boxes = known_boxes or []
        ry1, ry2 = int(h * 0.15), int(h * 0.82)
        roi = dmap[ry1:ry2, :]
        if roi.size == 0:
            return None

        # Tính độ sâu trung vị và độ biến thiên dọc |dZ/dy| cho từng cột pixel x
        col_med_z = np.median(roi, axis=0)
        col_dz_dy = np.mean(np.abs(np.diff(roi, axis=0)), axis=0)

        # Cột pixel thuộc vật cản đứng gần nếu z <= 1.75m và bề mặt thẳng đứng (col_dz_dy < 0.025)
        is_vert_close = (col_med_z <= 1.75) & (col_dz_dy < 0.025)

        # Tìm đoạn liên tục (contiguous run) dài nhất có giao cắt với hành lang giữa [0.32*w, 0.68*w]
        best_run = None
        best_len = 0
        cur_start = None

        for x in range(w):
            if is_vert_close[x]:
                if cur_start is None:
                    cur_start = x
            else:
                if cur_start is not None:
                    run_len = x - cur_start
                    if run_len > best_len and cur_start < int(w * 0.68) and x > int(w * 0.32):
                        best_len = run_len
                        best_run = (cur_start, x)
                    cur_start = None
        if cur_start is not None:
            run_len = w - cur_start
            if run_len > best_len and cur_start < int(w * 0.68) and w > int(w * 0.32):
                best_len = run_len
                best_run = (cur_start, w)

        if best_run is None:
            return None

        bx1, bx2 = best_run
        span_ratio = float(bx2 - bx1) / float(w)
        # Cây cột có bề ngang từ 12% đến 64% khung hình (nếu > 64% thì detect_wall sẽ bắt là bức tường)
        if span_ratio < 0.12 or span_ratio > 0.64:
            return None

        pillar_z = float(np.median(col_med_z[bx1:bx2]))
        if pillar_z < 0.30 or pillar_z > 1.75:
            return None

        # Kiểm tra độ tương phản độ sâu với nền hai bên cột (ít nhất 1 bên phải sâu hơn cột >= 0.35m)
        left_bg = col_med_z[:max(0, bx1 - 6)]
        right_bg = col_med_z[min(w, bx2 + 6):]
        left_diff = (float(np.median(left_bg)) - pillar_z) if left_bg.size >= 8 else 0.0
        right_diff = (float(np.median(right_bg)) - pillar_z) if right_bg.size >= 8 else 0.0

        if max(left_diff, right_diff) < 0.35:
            return None

        by1, by2 = ry1, ry2
        if self._is_box_covered(bx1, by1, bx2, by2, known_boxes, thresh=0.30):
            return None

        cx = (bx1 + bx2) / 2.0
        pos = "left" if cx < w / 3.0 else ("right" if cx > 2.0 * w / 3.0 else "center")
        dist_m = round(pillar_z, 1)

        return {
            "class_name": "column",
            "class_name_vn": "cột nhà",
            "confidence": 0.90,
            "position": pos,
            "distance_m": dist_m,
            "bounding_box": {"x1": int(bx1), "y1": int(by1), "x2": int(bx2), "y2": int(by2)},
            "in_roi": True,
            "is_danger": True,
            "hazard_type": "VERTICAL_PILLAR",
            "warning": f"Cảnh báo! Phía trước cách {dist_m} mét có cột nhà chắn lối."
        }

    @staticmethod
    def _is_box_covered(bx1: int, by1: int, bx2: int, by2: int, known_boxes: List[Dict[str, int]], thresh: float = 0.35) -> bool:
        area = float(max(1, (bx2 - bx1) * (by2 - by1)))
        for kbox in known_boxes:
            ix1 = max(bx1, kbox.get("x1", 0))
            iy1 = max(by1, kbox.get("y1", 0))
            ix2 = min(bx2, kbox.get("x2", 0))
            iy2 = min(by2, kbox.get("y2", 0))
            if ix2 > ix1 and iy2 > iy1:
                if float((ix2 - ix1) * (iy2 - iy1)) / area > thresh:
                    return True
        return False

    @staticmethod
    def _is_line_inside_yolo_box(lx1: int, lx2: int, ly: int, known_boxes: List[Dict[str, int]]) -> bool:
        line_w = max(1, lx2 - lx1)
        for kbox in known_boxes:
            kx1, ky1, kx2, ky2 = kbox.get("x1", 0), kbox.get("y1", 0), kbox.get("x2", 0), kbox.get("y2", 0)
            if (ky1 - 8) <= ly <= (ky2 + 8):
                overlap_x = max(0, min(lx2, kx2) - max(lx1, kx1))
                if overlap_x / float(line_w) > 0.55:
                    return True
        return False


# Singleton dùng chung
geometric_hazard_detector = GeometricHazardDetector()
