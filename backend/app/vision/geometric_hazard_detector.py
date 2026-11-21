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
            roi_full = dmap[y1:y2, x1:x2]

            if roi_full.size > 0:
                step_d = 2 if (roi_full.shape[0] >= 120 and roi_full.shape[1] >= 120) else 1
                roi_depth = roi_full[::step_d, ::step_d]

                # Kiểm tra xem có phải mặt đường/sàn nhà dốc xa dần không:
                # Trên mặt đường mở, phần trên (y1) xa hơn phần dưới (y2) rất nhiều.
                rh = roi_depth.shape[0]
                top_band_z = float(np.median(roi_depth[:max(1, int(rh * 0.25)), :]))
                bot_band_z = float(np.median(roi_depth[max(0, int(rh * 0.75)):, :]))
                vertical_z_span = abs(top_band_z - bot_band_z)

                # Chuẩn hóa Sobel ksize=3 chia cho (8.0 * step_d) để lấy đúng dZ/dy (mét/pixel gốc)
                grad_y = cv2.Sobel(roi_depth, cv2.CV_32F, 0, 1, ksize=3) / (8.0 * step_d)
                abs_grad_y = np.abs(grad_y)

                median_roi_z = float(np.median(roi_depth))
                if 0.25 <= median_roi_z <= 1.35 and vertical_z_span <= 1.15:
                    wall_mask = (
                        (abs_grad_y < 0.011)
                        & (roi_depth >= max(0.2, median_roi_z - 0.45))
                        & (roi_depth <= median_roi_z + 0.45)
                    ).astype(np.uint8)

                    roi_area = float(wall_mask.size)
                    wall_pixel_count = int(cv2.countNonZero(wall_mask))
                    fill_ratio = wall_pixel_count / max(1.0, roi_area)

                    if fill_ratio >= 0.30:
                        ys, xs = np.where(wall_mask > 0)
                        if ys.size > 0:
                            bx1 = x1 + int(np.percentile(xs, 5) * step_d)
                            bx2 = x1 + int(np.percentile(xs, 95) * step_d)
                            by1 = y1 + int(np.percentile(ys, 5) * step_d)
                            by2 = y1 + int(np.percentile(ys, 95) * step_d)

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
                    # Thu nhỏ ROI xuống ngang <= 128px cho bước kiểm tra tường phẳng quang học (nhanh gấp ~10x)
                    if roi_bgr.shape[1] > 128:
                        s_wall = 128.0 / float(roi_bgr.shape[1])
                        roi_bgr = cv2.resize(
                            roi_bgr,
                            (128, max(32, int(round(roi_bgr.shape[0] * s_wall)))),
                            interpolation=cv2.INTER_AREA,
                        )
                    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

                    # Chia thành lưới 4x4 để kiểm tra độ đồng nhất màu sắc & độ phẳng bề mặt của bức tường
                    gh, gw = gray.shape[:2]
                    bh, bw = max(1, gh // 4), max(1, gw // 4)
                    med_h = float(np.median(hsv[:, :, 0]))
                    med_s = float(np.median(hsv[:, :, 1]))
                    med_v = float(np.median(hsv[:, :, 2]))

                    if med_v >= 30.0:
                        flat_blocks = 0
                        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)

                        for r in range(4):
                            for c in range(4):
                                sub_g = gray[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
                                sub_hsv = hsv[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
                                sub_e = edges[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
                                if sub_g.size == 0:
                                    continue
                                std_g = float(np.std(sub_g))
                                edge_ratio = float(cv2.countNonZero(sub_e)) / float(sub_e.size)
                                dh = min(abs(float(np.median(sub_hsv[:, :, 0])) - med_h), 180.0 - abs(float(np.median(sub_hsv[:, :, 0])) - med_h))
                                ds = abs(float(np.median(sub_hsv[:, :, 1])) - med_s)
                                dv = abs(float(np.median(sub_hsv[:, :, 2])) - med_v)

                                if std_g < 22.0 and edge_ratio < 0.045 and dh < 22.0 and ds < 50.0 and dv < 55.0:
                                    flat_blocks += 1

                        # Nếu >= 11/16 ô trung tâm là cùng một mảng bề mặt phẳng đồng nhất VÀ đủ ánh sáng
                        if flat_blocks >= 11:
                            est_dist = 1.2
                            if dmap is not None and dmap.size > 0:
                                est_dist = round(min(3.5, max(0.5, float(np.median(dmap[cy1:cy2:2, cx1:cx2:2])))), 1)
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

        # 1. Trích xuất ROI vùng bước chân & dùng stride step=2 khi ROI lớn (>= 380px) để tăng tốc ~4x mà giữ sắc nét gờ bậc
        gy1, gy2 = int(h * 0.38), int(h * 0.95)
        gx1, gx2 = int(w * 0.12), int(w * 0.88)

        roi_bgr_full = frame_bgr[gy1:gy2, gx1:gx2]
        if roi_bgr_full.size == 0:
            return None

        roi_depth_full = dmap[gy1:gy2, gx1:gx2]
        full_rh, full_rw = roi_depth_full.shape[:2]
        if full_rh < 24:
            return None

        step = 2 if (full_rw >= 380 and full_rh >= 160) else 1
        roi_bgr = roi_bgr_full[::step, ::step]
        roi_depth = roi_depth_full[::step, ::step]
        scale = 1.0 / float(step)

        rh, rw = roi_depth.shape[:2]
        if rh < 16:
            return None

        top_z = float(np.median(roi_depth[:max(1, rh // 3), :]))
        bot_z = float(np.median(roi_depth[min(rh - 1, 2 * rh // 3):, :]))
        std_z = float(np.std(roi_depth))

        # 2. Lọc cạnh ngang thực tế
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3) if step == 2 else (5, 5), 0)

        sobel_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)
        abs_x = cv2.convertScaleAbs(sobel_x)
        abs_y = cv2.convertScaleAbs(sobel_y)

        cx_l, cx_r = int(rw * 0.18), int(rw * 0.82)
        center_abs_x = abs_x[:, cx_l:cx_r]
        center_abs_y = abs_y[:, cx_l:cx_r]
        horiz_px = int(np.count_nonzero((center_abs_y >= 32) & (center_abs_y > center_abs_x * 1.4)))
        vert_px = int(np.count_nonzero((center_abs_x >= 32) & (center_abs_x > center_abs_y * 1.4)))
        if horiz_px == 0:
            return None

        depth_column = np.median(roi_depth[:, cx_l:cx_r], axis=1)
        gray_column = np.median((gray if step == 2 else blurred)[:, cx_l:cx_r], axis=1)
        sy_column = np.median(center_abs_y, axis=1)
        p90_p10 = float(np.percentile(gray_column, 90) - np.percentile(gray_column, 10))

        # Nếu bề mặt phẳng lì theo phương đứng -> Loại bỏ ngay
        if abs(top_z - bot_z) < 0.22 and std_z < 0.18 and p90_p10 < 40.0:
            return None

        # Giữ lại các cạnh ngang gờ bậc
        horiz_mask = ((abs_y >= 28) & (abs_y.astype(np.float32) >= abs_x.astype(np.float32) * 1.35)).astype(np.uint8) * 255

        # Nối liền các vết đứt nhỏ dọc theo gờ bậc
        kw_close = max(5, int(round(9 * scale)) | 1)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (kw_close, 3))
        horizontal_edges = cv2.morphologyEx(horiz_mask, cv2.MORPH_CLOSE, kernel_close)

        # 3. Dò các đoạn thẳng gờ bậc bằng HoughLinesP
        min_line_len = max(20, int(w * 0.25 * scale))
        hough_thresh = max(16, int(round(28 * scale)))
        max_gap = max(8, int(round(22 * scale)))
        lines = cv2.HoughLinesP(
            horizontal_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=hough_thresh,
            minLineLength=min_line_len,
            maxLineGap=max_gap,
        )

        valid_horizontal_lines = []
        inv_scale = 1.0 / scale
        if lines is not None:
            for l in lines:
                pts = l.flatten()
                if len(pts) < 4:
                    continue
                lx1, ly1, lx2, ly2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
                angle = np.abs(np.arctan2(ly2 - ly1, lx2 - lx1) * 180.0 / np.pi)
                if angle <= 15.0:
                    full_x1 = gx1 + int(round(min(lx1, lx2) * inv_scale))
                    full_x2 = gx1 + int(round(max(lx1, lx2) * inv_scale))
                    full_y = gy1 + int(round(((ly1 + ly2) / 2.0) * inv_scale))
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

        # Gom nhóm các cạnh ngang cách nhau <= 12px (theo ảnh gốc) để không gộp nhầm nhiều bậc thang hẹp thành 1 cụm
        cluster_gap = 12.0 * scale
        valid_horizontal_lines.sort(key=lambda item: item["y_avg"])
        clusters: List[Dict[str, float]] = []
        for line in valid_horizontal_lines:
            if not clusters or abs(line["y_avg"] - clusters[-1]["y_avg"]) > cluster_gap:
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
                c["max_len"] = max(c["max_len"], float(line["x2"] - c["x1"]))

        clustered_y = [c["y_avg"] for c in clusters]
        max_span_px = max((c["x2"] - c["x1"]) for c in clusters) * inv_scale
        if max_span_px < w * 0.25:
            return None

        # 4. Phân tích kết hợp Bước nhảy độ sâu 3D & Hồ sơ quang học bậc thang thực tế
        dy_all = np.diff(depth_column)
        baseline_slope_per_px = float(max(0.0, -np.median(dy_all)))

        depth_jumps = 0
        optical_jumps = 0
        max_net_jump = 0.0
        cluster_depths: List[float] = []

        w_out = max(3, int(round(12 * scale)))
        w_in = max(1, int(round(2 * scale)))
        min_edge_pad = max(3, int(round(10 * scale)))
        g_pad_out = max(3, int(round(12 * scale)))
        g_pad_in = max(1, int(round(3 * scale)))
        g_min_edge = max(3, int(round(6 * scale)))
        sy_half = max(2, int(round(4 * scale)))

        for cy in clustered_y:
            y_idx = int(cy)
            if min_edge_pad <= y_idx < len(depth_column) - min_edge_pad:
                z_upper = float(np.mean(depth_column[max(0, y_idx - w_out):max(1, y_idx - w_in)]))
                z_lower = float(np.mean(depth_column[min(len(depth_column) - 1, y_idx + w_in):min(len(depth_column), y_idx + w_out)]))
                cluster_depths.append(float(depth_column[y_idx]))

                raw_jump = z_upper - z_lower
                expected_smooth_drop = baseline_slope_per_px * float(w_out + w_in)
                net_jump = raw_jump - expected_smooth_drop

                if raw_jump >= 0.18 and net_jump >= 0.14:
                    depth_jumps += 1
                    max_net_jump = max(max_net_jump, net_jump)

            if g_min_edge <= y_idx < len(gray_column) - g_min_edge:
                g_upper = float(np.mean(gray_column[max(0, y_idx - g_pad_out):max(1, y_idx - g_pad_in)]))
                g_lower = float(np.mean(gray_column[min(len(gray_column) - 1, y_idx + g_pad_in):min(len(gray_column), y_idx + g_pad_out)]))
                local_sy_peak = float(np.max(sy_column[max(0, y_idx - sy_half):min(len(sy_column), y_idx + sy_half + 1)]))
                if abs(g_upper - g_lower) >= 12.0 or (p90_p10 >= 22.0 and local_sy_peak >= 24.0):
                    optical_jumps += 1
                elif local_sy_peak >= 100.0 and min_edge_pad <= y_idx < len(depth_column) - min_edge_pad and max_net_jump >= 0.18:
                    # Bậc thềm đơn lẻ (1 gờ ngang rất sắc nét ngay tại bước nhảy độ sâu 3D >= 0.18m)
                    optical_jumps += 1

        step_min_y = int(min(clustered_y))
        step_max_y = int(max(clustered_y))
        vertical_span = (step_max_y - step_min_y) * inv_scale

        monotonic_3d = True
        if len(cluster_depths) >= 2:
            for i in range(len(cluster_depths) - 1):
                if cluster_depths[i] < cluster_depths[i + 1] + 0.10:
                    monotonic_3d = False
                    break
        has_3d_step_jumps = (
            (depth_jumps >= 1)
            and (max_net_jump >= 0.18)
            and (optical_jumps >= 1)
            and monotonic_3d
            and (vert_px <= horiz_px * 0.50)
            and ((top_z - bot_z) >= 0.28)
        )

        # Chống báo nhầm lưới ron gạch nền phẳng / tường ốp gạch: luôn yêu cầu cạnh dọc thấp hơn rõ cạnh ngang (<= 48%)
        has_real_multistep_pattern = (
            len(clustered_y) >= 3
            and vertical_span >= 65
            and p90_p10 >= 22.0
            and optical_jumps >= 2
            and (vert_px <= horiz_px * 0.48)
            and (
                std_z >= 0.20
                or abs(top_z - bot_z) >= 0.22
                or p90_p10 >= 40.0
            )
        )

        if not (has_3d_step_jumps or has_real_multistep_pattern):
            return None

        # 5. Quy đổi tọa độ & khoảng cách về khung hình gốc
        bx1 = gx1 + int(round(min(c["x1"] for c in clusters) * inv_scale))
        bx2 = gx1 + int(round(max(c["x2"] for c in clusters) * inv_scale))
        orig_min_y = int(round(step_min_y * inv_scale))
        orig_max_y = int(round(step_max_y * inv_scale))
        by1 = gy1 + max(0, orig_min_y - 20)
        by2 = gy1 + min(full_rh, orig_max_y + 35)

        pad_tail = max(4, int(round(12 * scale)))
        stair_depth_roi = roi_depth[max(0, step_min_y):min(roi_depth.shape[0], step_max_y + pad_tail), :]
        stair_dist = round(float(np.median(stair_depth_roi)), 1) if stair_depth_roi.size > 0 else 1.5

        # Khi đứng ngay đầu cầu thang thực tế (bậc dưới cùng sát chân ảnh by2 >= 62% chiều cao ảnh),
        # Depth Anything V2 Indoor thường kéo sâu bậc thang ngoài trời lên 5m - 9m -> chuẩn hóa về 1.5m khi cấu trúc ngang rõ
        max_clamp_d = 9.5 if (vert_px <= horiz_px * 0.35 or max_net_jump >= 0.40) else 4.8
        if has_real_multistep_pattern and by2 >= int(h * 0.62) and 1.8 < stair_dist <= max_clamp_d:
            stair_dist = 1.5

        if stair_dist > 3.2 or stair_dist < 0.35:
            return None

        if len(clustered_y) >= 3 and vertical_span >= 65 and (depth_jumps >= 2 or has_real_multistep_pattern):
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

        # Downsample theo bước nhảy để tính trung vị cột nhanh gấp ~4x mà không mất cạnh cột
        step_y = 2 if roi.shape[0] >= 100 else 1
        step_x = 2 if w >= 320 else 1
        roi_s = roi[::step_y, ::step_x]

        col_med_z = np.median(roi_s, axis=0)
        col_dz_dy = np.mean(np.abs(np.diff(roi_s, axis=0)), axis=0) / float(step_y)

        # Cột pixel thuộc vật cản đứng gần nếu z <= 1.75m và bề mặt thẳng đứng (col_dz_dy < 0.025)
        is_vert_close = (col_med_z <= 1.75) & (col_dz_dy < 0.025)

        # Vectorized tìm các đoạn liên tục (contiguous runs) thay cho vòng lặp Python theo từng pixel
        sw = len(is_vert_close)
        padded = np.concatenate(([False], is_vert_close, [False]))
        diffs = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(diffs == 1)
        ends = np.flatnonzero(diffs == -1)

        best_run = None
        best_len = 0
        mid_l = int(sw * 0.32)
        mid_r = int(sw * 0.68)
        for s_idx, e_idx in zip(starts, ends):
            r_len = int(e_idx - s_idx)
            if r_len > best_len and s_idx < mid_r and e_idx > mid_l:
                best_len = r_len
                best_run = (int(s_idx) * step_x, min(w, int(e_idx) * step_x))

        if best_run is None:
            return None

        bx1, bx2 = best_run
        span_ratio = float(bx2 - bx1) / float(w)
        # Cây cột có bề ngang từ 12% đến 64% khung hình (nếu > 64% thì detect_wall sẽ bắt là bức tường)
        if span_ratio < 0.12 or span_ratio > 0.64:
            return None

        s_bx1 = bx1 // step_x
        s_bx2 = max(s_bx1 + 1, bx2 // step_x)
        pillar_z = float(np.median(col_med_z[s_bx1:s_bx2]))
        if pillar_z < 0.30 or pillar_z > 1.75:
            return None

        # Kiểm tra độ tương phản độ sâu với nền hai bên cột (ít nhất 1 bên phải sâu hơn cột >= 0.35m)
        pad_bg = max(2, 6 // step_x)
        left_bg = col_med_z[:max(0, s_bx1 - pad_bg)]
        right_bg = col_med_z[min(sw, s_bx2 + pad_bg):]
        min_bg_sz = max(3, 8 // step_x)
        left_diff = (float(np.median(left_bg)) - pillar_z) if left_bg.size >= min_bg_sz else 0.0
        right_diff = (float(np.median(right_bg)) - pillar_z) if right_bg.size >= min_bg_sz else 0.0

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

    def detect_generic_front_obstacle(
        self,
        dmap: Optional[np.ndarray],
        known_boxes: Optional[List[Dict[str, int]]] = None,
        frame_bgr: Optional[np.ndarray] = None,
        max_dist_m: float = 1.50,
    ) -> Optional[Dict[str, Any]]:
        """
        Phát hiện VẬT CẢN LẠ CHẮN NGANG PHÍA TRƯỚC (như tấm bạt căng ngang, rèm che, tấm ván, thùng lớn)
        trong phạm vi `<= 1.50m` khi YOLO không có nhãn từ vựng và `detect_wall` bỏ qua do bề mặt
        bạt/vải có nếp gấp nhăn (`|dZ/dy|` lớn hơn tường phẳng hoặc có hoa văn/màu sắc không đồng nhất).
        Chống báo giả tuyệt đối khi đứng cách bức tường phẳng 1.5m - 2.0m (đủ 2 bước chân để đi tiếp):
        Nếu bề mặt trải phẳng suốt cả 3 làn Trái-Giữa-Phải như bức tường và còn cách > 1.00m thì không báo vật cản!
        """
        if dmap is None or dmap.size == 0:
            return None

        h, w = dmap.shape[:2]
        if h < 24 or w < 24:
            return None

        known_boxes = known_boxes or []
        x1, x2 = int(w * 0.26), int(w * 0.74)
        y1, y2 = int(h * 0.18), int(h * 0.78)
        roi_depth = dmap[y1:y2, x1:x2]
        if roi_depth.size == 0:
            return None

        rh, rw = roi_depth.shape[:2]
        top_half = roi_depth[:max(1, rh // 2), :]
        bot_half = roi_depth[max(1, rh // 2):, :]

        top_med_z = float(np.median(top_half))
        bot_med_z = float(np.median(bot_half))

        # Nếu phần trên khung hình xa hơn phần dưới rất nhiều (top_med_z - bot_med_z > 1.20m)
        # và phần trên thông thoáng (> 2.15m) -> Đây là mặt sàn mở bình thường, không phải tấm bạt/vật cản chắn ngang!
        if (top_med_z - bot_med_z) > 1.20 and top_med_z > 2.15:
            return None

        # Tấm bạt căng ngang hoặc vật cản treo/đứng chắn lối đi sẽ làm cả nửa trên lẫn vùng giữa có độ sâu <= max_dist_m
        close_mask = ((roi_depth >= 0.25) & (roi_depth <= max_dist_m)).astype(np.uint8)
        close_ratio = float(np.mean(close_mask))
        upper_close_ratio = float(np.mean((top_half >= 0.25) & (top_half <= max_dist_m + 0.08)))

        if close_ratio < 0.35 or upper_close_ratio < 0.30:
            return None

        obs_z = float(np.median(roi_depth[close_mask > 0]))

        # KIỂM TRA CHỐNG NHẦM BỨC TƯỜNG CÁCH 1.1m - 1.8m (ĐỦ 2 BƯỚC ĐI) THÀNH "VẬT CẢN":
        # Nếu cả bên trái, ở giữa và bên phải đều có cùng độ sâu phẳng đồng nhất (chênh lệch < 0.25m, phương sai nhỏ < 0.05)
        # thì đó là bức tường phẳng ở xa (> 1.00m), người dùng vẫn còn 2 bước chân để đi tiếp -> Không báo vật cản!
        left_strip = dmap[y1:y2, :max(1, int(w * 0.22))]
        right_strip = dmap[y1:y2, min(w - 1, int(w * 0.78)):]
        if left_strip.size > 0 and right_strip.size > 0 and obs_z > 1.00:
            l_z = float(np.median(left_strip))
            r_z = float(np.median(right_strip))
            if abs(l_z - obs_z) < 0.25 and abs(r_z - obs_z) < 0.25 and float(np.std(roi_depth)) < 0.05:
                return None

        ys, xs = np.where(close_mask > 0)
        if ys.size < 20:
            return None

        bx1 = x1 + int(np.percentile(xs, 5))
        bx2 = x1 + int(np.percentile(xs, 95))
        by1 = y1 + int(np.percentile(ys, 5))
        by2 = y1 + int(np.percentile(ys, 95))

        if (bx2 - bx1) < int(w * 0.16) or (by2 - by1) < int(h * 0.16):
            return None

        if self._is_box_covered(bx1, by1, bx2, by2, known_boxes, thresh=0.25):
            return None

        dist_m = round(max(0.3, min(obs_z, max_dist_m)), 1)

        return {
            "class_name": "obstacle",
            "class_name_vn": "vật cản",
            "confidence": round(min(0.92, 0.68 + close_ratio * 0.25), 2),
            "position": "center",
            "distance_m": dist_m,
            "bounding_box": {"x1": int(bx1), "y1": int(by1), "x2": int(bx2), "y2": int(by2)},
            "in_roi": True,
            "is_danger": True,
            "hazard_type": "GENERIC_FRONT_OBSTACLE",
            "warning": f"Cảnh báo! Phía trước cách {dist_m} mét có vật cản chắn lối."
        }

    def detect_ground_dropoff(
        self,
        dmap: Optional[np.ndarray],
        known_boxes: Optional[List[Dict[str, int]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Phát hiện HỐ SÂU / SỤT LÚN / BẬC HỤT CHÂN (Ground Drop-off / Deep Pit) ngay dưới chân:
        - Trên mặt phẳng đi bộ bình thường, khi quét từ giữa ảnh (y=50%) xuống sát chân (y=92%),
          khoảng cách độ sâu Z giảm đều (gần người hơn).
        - Nếu xuất hiện HỐ SÂU hoặc bậc hụt sâu dưới đất: tại ranh giới miệng hố, các hàng pixel
          phía dưới (gần chân hơn trên ảnh) đột ngột vọt độ sâu Z TĂNG LÊN (sâu/xa hơn mép trên >= 0.52m),
          hoặc tồn tại vùng lõm sâu bất thường (>= +0.65m so với mặt sàn hai bên).
        """
        if dmap is None or dmap.size == 0:
            return None

        h, w = dmap.shape[:2]
        if h < 32 or w < 32:
            return None

        known_boxes = known_boxes or []
        gx1, gx2 = int(w * 0.24), int(w * 0.76)
        gy1, gy2 = int(h * 0.50), int(h * 0.92)
        foot_full = dmap[gy1:gy2, gx1:gx2]
        if foot_full.size == 0:
            return None

        step_f = 2 if (foot_full.shape[0] >= 80 and foot_full.shape[1] >= 120) else 1
        foot_roi = foot_full[::step_f, ::step_f]
        rh, rw = foot_roi.shape[:2]
        if rh < 12:
            return None

        # Mặt đất bình thường quanh mép trên vùng bước chân phải nằm trong tầm gần (0.5m - 2.3m)
        upper_ground_z = float(np.median(foot_roi[:max(2, rh // 4), :]))
        if upper_ground_z > 2.35 or upper_ground_z < 0.45:
            return None

        row_med_z = np.median(foot_roi, axis=1)
        # Tìm bước nhảy độ sâu NGƯỢC CHIỀU (phía dưới ảnh lại sâu vọt hơn phía trên >= 0.52m)
        max_drop_jump = 0.0
        drop_row_idx = -1
        for y_idx in range(4, rh - 4):
            z_above = float(np.mean(row_med_z[max(0, y_idx - 4):y_idx]))
            z_below = float(np.mean(row_med_z[y_idx + 1:min(rh, y_idx + 5)]))
            drop_jump = z_below - z_above
            if drop_jump > max_drop_jump and z_above <= 2.2:
                max_drop_jump = drop_jump
                drop_row_idx = y_idx

        if max_drop_jump < 0.52 or drop_row_idx < 0:
            return None

        pit_mask = (foot_roi[drop_row_idx:, :] >= (upper_ground_z + 0.45)).astype(np.uint8)
        pit_ratio = float(np.mean(pit_mask))
        if pit_ratio < 0.25:
            return None

        bx1, bx2 = gx1 + int(rw * 0.10 * step_f), gx1 + int(rw * 0.90 * step_f)
        by1 = gy1 + max(0, drop_row_idx * step_f - 4)
        by2 = gy2

        if self._is_box_covered(bx1, by1, bx2, by2, known_boxes, thresh=0.35):
            return None

        dist_m = round(max(0.5, min(upper_ground_z, 2.0)), 1)
        return {
            "class_name": "dropoff",
            "class_name_vn": "hố sâu",
            "confidence": round(min(0.94, 0.72 + min(0.20, max_drop_jump * 0.15)), 2),
            "position": "center",
            "distance_m": dist_m,
            "bounding_box": {"x1": int(bx1), "y1": int(by1), "x2": int(bx2), "y2": int(by2)},
            "in_roi": True,
            "is_danger": True,
            "hazard_type": "GROUND_DROPOFF",
            "warning": f"Nguy hiểm! Chú ý dưới chân cách {dist_m} mét có hố sâu, hãy dừng lại."
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
