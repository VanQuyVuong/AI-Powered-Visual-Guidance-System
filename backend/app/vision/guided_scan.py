"""
Guided Scan - Bộ điều khiển giọng nói dẫn đường tinh gọn cho người khiếm thị.

Quy tắc đơn giản, dứt khoát theo yêu cầu người dùng:
1. Phía trước an toàn:
   -> "Phía trước an toàn, đi thẳng."
2. Phía trước có vật cản hoặc tường gần (<= 1.3m):
   - Bên trái trống  -> "Phía trước có [tên], hãy rẽ sang trái."
   - Bên phải trống -> "Phía trước có [tên], hãy rẽ sang phải."
   - Cả 2 bên đều vướng -> "Phía trước có [tên], cả hai bên đều vướng, hãy quay đầu."
3. Dưới chân có bậc thềm / cầu thang:
   -> "Chú ý dưới chân, có [bậc thềm/cầu thang], hãy bước chậm."
4. Chống nói lặp liên tục: Đọc 1 lần dứt khoát, giữ yên lặng tránh gây ồn/loạn tai.
"""

from typing import Optional

# Khoảng cách nguy hiểm (m)
WALL_DANGER_M = 1.00        # Bức tường chỉ báo rẽ/quay đầu khi tiến sát <= 1.0m (còn cách 1.1m - 1.8m tức 2 bước chân vẫn cho đi tiếp)
OBSTACLE_DANGER_M = 2.25    # Vật cản ở hành lang an toàn / làn giữa <= 2.25m (chặn triệt để báo nhầm ghế ở xa 3m - 5.5m)
SIDE_OBSTACLE_M = 1.65      # Vật cản ở 2 bên <= 1.65m

ALERT_REPEAT_INTERVAL_S = 6.0   # Nhắc lại sau 6s nếu vẫn đứng trước vật cản
CLEAR_REPEAT_INTERVAL_S = 10.0  # Nhắc lại "an toàn, đi thẳng" sau 10s

STEP_WORDS = ("bậc thềm", "bậc tam cấp", "cầu thang", "bậc thang", "ổ gà", "hố sụt", "hố sâu")
GROUND_HAZARD_CLASSES = {"stairs", "step", "curb", "doorstep", "threshold", "pothole", "dropoff"}
WALL_CLASSES = {"wall"}


def _initial_state() -> dict:
    return {
        "phase": "PATROL",
        "last_text": "",
        "last_priority": "LOW",
        "last_time": 0.0,
        "last_alert_sig": "",
        "last_dist_m": 9.9,
        "clear_frames": 0,
        "blocked_frames": 0,
        "history": [],  # dùng bởi pipeline (chống rung tay)
    }


voice_state = _initial_state()


def reset_voice_state() -> None:
    voice_state.clear()
    voice_state.update(_initial_state())


def _get(o, key, default=None):
    if isinstance(o, dict):
        return o.get(key, default)
    return getattr(o, key, default)


def _box(o):
    b = _get(o, "bounding_box")
    if b is None:
        return None
    if isinstance(b, dict):
        return float(b.get("x1", 0)), float(b.get("x2", 0))
    return float(getattr(b, "x1", 0)), float(getattr(b, "x2", 0))


def assess_path(cur_text: str, all_objects: list, depth_lanes: Optional[dict], img_width: int) -> dict:
    """
    Đánh giá lối đi thời gian thực:
    Trả về:
      - status: CLEAR | ALLEY_CLEAR | ALLEY_DRIFT_LEFT | ALLEY_DRIFT_RIGHT | TURN_LEFT | TURN_RIGHT | TURN_AROUND | STEPS
      - name: tên vật cản (vật cản, bức tường, cột, hố sâu,...)
      - min_dist: khoảng cách vật cản gần nhất (m)
    """
    lower = (cur_text or "").lower()
    W = float(max(1, img_width))
    objs = list(all_objects or [])

    lanes = (depth_lanes or {}).get("lanes", {}) or {}
    has_depth = bool(lanes)

    def lane(name):
        return lanes.get(name, {}) if has_depth else {}

    l_blk = bool(lane("left").get("is_blocked", False))
    c_blk = bool(lane("center").get("is_blocked", False))
    r_blk = bool(lane("right").get("is_blocked", False))
    l_med = float(lane("left").get("median_m", 3.0) or 3.0)
    c_med = float(lane("center").get("median_m", 3.0) or 3.0)
    r_med = float(lane("right").get("median_m", 3.0) or 3.0)

    def dist(o):
        d = _get(o, "distance_m")
        return float(d) if d is not None else 4.0

    def is_in_path(o) -> bool:
        cname = str(_get(o, "class_name", "")).lower()
        if cname in WALL_CLASSES:
            return False
        d = dist(o)
        # BẮT BUỘC: Bất kỳ vật cản nào ở xa > OBSTACLE_DANGER_M (2.25m, ví dụ ghế cách 5.5m)
        # tuyệt đối KHÔNG coi là chắn đường!
        if d > OBSTACLE_DANGER_M:
            return False

        # 1. Hộp bao xâm lấn vào làn giữa (x1 < 0.65*W và x2 > 0.35*W)
        b = _box(o)
        if b is not None:
            bx1, bx2 = b
            if bx1 < 0.65 * W and bx2 > 0.35 * W:
                return True

        # 2. Lọt vào hình thang hành lang an toàn 3D (ROI)
        if _get(o, "in_roi") is True:
            return True

        # 3. Vị trí ở làn giữa
        if _get(o, "position") == "center":
            return True

        return False

    # 1. BẬC THỀM / CẦU THANG / Ổ GÀ / HỐ SÂU DƯỚI CHÂN (Kiểm tra trực tiếp cả trong all_objects lẫn cur_text
    # để không bao giờ bị vật cản 2 bên che khuất cảnh báo dưới chân!)
    step_objs = [
        o for o in objs
        if str(_get(o, "class_name", "")).lower() in GROUND_HAZARD_CLASSES and dist(o) <= 2.40
    ]
    is_steps = bool(step_objs) or any(k in lower for k in STEP_WORDS)
    if is_steps:
        if step_objs:
            nearest_step = min(step_objs, key=dist)
            raw_cls = str(_get(nearest_step, "class_name", "")).lower()
            step_name = _get(nearest_step, "class_name_vn") or (
                "hố sâu" if raw_cls == "dropoff" else ("cầu thang" if raw_cls == "stairs" else "bậc thềm")
            )
            step_d = dist(nearest_step)
        elif "hố sâu" in lower:
            step_name = "hố sâu"
            step_d = 1.4
        elif "ổ gà" in lower or "hố sụt" in lower:
            step_name = "ổ gà"
            step_d = 1.5
        elif "cầu thang" in lower or "bậc thang" in lower:
            step_name = "cầu thang"
            step_d = 1.5
        else:
            step_name = "bậc thềm"
            step_d = 1.5
        return {"status": "STEPS", "name": step_name, "min_dist": step_d}

    # 2. KIỂM TRA BỨC TƯỜNG CHẮN TRƯỚC MẶT (Chỉ nguy hiểm khi tường ở trung tâm tiến sát <= WALL_DANGER_M)
    wall_objs = [
        o for o in objs
        if str(_get(o, "class_name", "")).lower() in WALL_CLASSES
        and _get(o, "position", "center") == "center"
        and dist(o) <= WALL_DANGER_M
    ]
    is_wall_near = bool(wall_objs) or (("bức tường" in lower) and (c_med <= WALL_DANGER_M))

    # 3. VẬT CẢN TRUNG TÂM / VÙNG HÀNH LANG AN TOÀN (CENTER / PATH)
    center_objs = [o for o in objs if is_in_path(o)]

    # Nếu chỉ có các vật thể ở rất xa (> 2.25m, ví dụ ghế cách 5.5m) hoặc tường còn cách > 1.0m (đủ 2 bước chân)
    # thì không bao giờ coi là đường bị chắn
    has_only_far_center_objs = bool(objs) and not center_objs and all(dist(o) > OBSTACLE_DANGER_M for o in objs)
    depth_strictly_blocked = c_blk and (c_med <= 0.95 if has_only_far_center_objs else c_med <= 1.00)

    center_blocked = is_wall_near or bool(center_objs) or depth_strictly_blocked

    # 4. NẾU LÀN GIỮA THÔNG THOÁNG -> KIỂM TRA CHẾ ĐỘ ĐI TRONG HẺM (ALLEY MODE) HOẶC ĐI THẲNG AN TOÀN
    if not center_blocked:
        # Kiểm tra xem hai bên có phải là 2 bức tường hẹp của con hẻm / hành lang tường không
        side_non_wall_objs = [
            o for o in objs
            if _get(o, "position") in ("left", "right")
            and str(_get(o, "class_name", "")).lower() not in WALL_CLASSES
            and dist(o) <= SIDE_OBSTACLE_M
        ]
        left_wall_obj = any(
            str(_get(o, "class_name", "")).lower() in WALL_CLASSES and _get(o, "position") == "left" and dist(o) <= 1.65
            for o in objs
        )
        right_wall_obj = any(
            str(_get(o, "class_name", "")).lower() in WALL_CLASSES and _get(o, "position") == "right" and dist(o) <= 1.65
            for o in objs
        )
        is_alley_walls = (
            (not side_non_wall_objs)
            and (
                (has_depth and l_med <= 1.45 and r_med <= 1.45 and c_med >= 1.95)
                or (left_wall_obj and right_wall_obj)
            )
        )
        if is_alley_walls:
            if has_depth and l_med <= 0.72 and r_med > 0.80:
                return {"status": "ALLEY_DRIFT_LEFT", "name": "tường bên trái", "min_dist": l_med}
            if has_depth and r_med <= 0.72 and l_med > 0.80:
                return {"status": "ALLEY_DRIFT_RIGHT", "name": "tường bên phải", "min_dist": r_med}
            return {"status": "ALLEY_CLEAR", "name": "tường dài hai bên", "min_dist": c_med}

        return {"status": "CLEAR", "name": "an toàn", "min_dist": 9.9}

    # 5. XÁC ĐỊNH TÊN VẬT CẢN CHẮN TRƯỚC MẶT (ƯU TIÊN TÊN VẬT THỂ CỤ THỂ)
    min_d = c_med
    if len(center_objs) >= 2:
        nearest = min(center_objs, key=dist)
        min_d = dist(nearest)
        unique_names = {(_get(o, "class_name_vn") or _get(o, "class_name") or "vật cản") for o in center_objs}
        if len(unique_names) == 1:
            only_name = next(iter(unique_names))
            name = f"nhiều {only_name}" if only_name != "vật cản" else "nhiều vật cản"
        else:
            name = "nhiều vật cản"
    elif center_objs:
        nearest = min(center_objs, key=dist)
        min_d = dist(nearest)
        name = _get(nearest, "class_name_vn") or _get(nearest, "class_name") or "vật cản"
    elif is_wall_near:
        min_d = min((dist(w) for w in wall_objs), default=c_med)
        name = "bức tường"
    else:
        name = "vật cản"

    # 6. ĐÁNH GIÁ 2 BÊN TRÁI & PHẢI
    left_near = [
        o for o in objs
        if _get(o, "position") == "left" and dist(o) <= SIDE_OBSTACLE_M and o not in center_objs
    ]
    right_near = [
        o for o in objs
        if _get(o, "position") == "right" and dist(o) <= SIDE_OBSTACLE_M and o not in center_objs
    ]

    # Kiểm tra khoảng trống vật lý trong khung hình
    boxes = [b for b in (_box(o) for o in center_objs) if b is not None]
    if boxes:
        ux1 = min(b[0] for b in boxes)
        ux2 = max(b[1] for b in boxes)
        left_gap = ux1 / W
        right_gap = (W - ux2) / W
        is_wide_obstacle = (ux2 - ux1) / W >= 0.72
    else:
        left_gap = right_gap = 0.33
        is_wide_obstacle = False

    # Độ vướng của bên trái / bên phải
    left_blocked = bool(left_near) or l_blk or (left_gap < 0.18) or (has_depth and l_med <= 1.25)
    right_blocked = bool(right_near) or r_blk or (right_gap < 0.18) or (has_depth and r_med <= 1.25)

    # Nếu là bức tường áp sát hoặc vật cản quá rộng che hết cả 2 bên
    if is_wall_near:
        if not (has_depth and l_med >= 1.9):
            left_blocked = True
        if not (has_depth and r_med >= 1.9):
            right_blocked = True
    elif is_wide_obstacle or (has_depth and l_blk and r_blk):
        left_blocked = right_blocked = True

    # 7. QUYẾT ĐỊNH ĐƠN GIẢN: RẼ TRÁI / RẼ PHẢI / QUAY ĐẦU
    if left_blocked and right_blocked:
        return {"status": "TURN_AROUND", "name": name, "min_dist": min_d}
    elif not left_blocked and right_blocked:
        return {"status": "TURN_LEFT", "name": name, "min_dist": min_d}
    elif left_blocked and not right_blocked:
        return {"status": "TURN_RIGHT", "name": name, "min_dist": min_d}
    else:
        # Cả hai bên đều còn khoảng trống: Chọn bên thoáng hơn
        if abs(left_gap - right_gap) >= 0.06:
            status = "TURN_LEFT" if left_gap > right_gap else "TURN_RIGHT"
        else:
            status = "TURN_LEFT" if l_med >= r_med else "TURN_RIGHT"
        return {"status": status, "name": name, "min_dist": min_d}


def update_spatial_scan_and_voice(
    cur_text: str,
    cur_priority: str,
    all_objects: list,
    depth_lanes: Optional[dict],
    img_width: int,
    now: float,
) -> tuple:
    """
    Tạo câu thông báo súc tích và điều khiển nhịp đọc tự nhiên, chống ồn:
    Trả về: (text, priority, should_speak_now)
    """
    st = voice_state
    a = assess_path(cur_text, all_objects, depth_lanes, img_width)
    status = a["status"]
    name = a["name"]
    min_dist = float(a.get("min_dist", 9.9))

    is_clear_like = status in ("CLEAR", "ALLEY_CLEAR")
    st["clear_frames"] = st["clear_frames"] + 1 if is_clear_like else 0
    st["blocked_frames"] = st["blocked_frames"] + 1 if not is_clear_like else 0

    since = (now - st["last_time"]) if st["last_time"] > 0 else float("inf")
    last_sig = st["last_alert_sig"]
    last_dist = float(st.get("last_dist_m", 9.9))

    target_text = ""
    target_prio = "HIGH"
    sig = f"{status}:{name}"

    if status == "STEPS":
        if name == "hố sâu":
            target_text = "Chú ý dưới chân, có hố sâu, hãy dừng lại."
            target_prio = "CRITICAL"
        else:
            target_text = f"Chú ý dưới chân, có {name}, hãy bước chậm."
            target_prio = "HIGH"
    elif status == "TURN_AROUND":
        target_text = f"Phía trước có {name}, cả hai bên đều vướng, hãy quay đầu."
        target_prio = "CRITICAL"
    elif status == "TURN_LEFT":
        target_text = f"Phía trước có {name}, hãy rẽ sang trái."
        target_prio = "HIGH"
    elif status == "TURN_RIGHT":
        target_text = f"Phía trước có {name}, hãy rẽ sang phải."
        target_prio = "HIGH"
    elif status == "ALLEY_DRIFT_LEFT":
        target_text = "Đang đi sát tường bên trái, hãy chỉnh nhẹ sang phải rồi đi thẳng."
        target_prio = "HIGH"
        sig = "ALLEY_DRIFT_LEFT"
    elif status == "ALLEY_DRIFT_RIGHT":
        target_text = "Đang đi sát tường bên phải, hãy chỉnh nhẹ sang trái rồi đi thẳng."
        target_prio = "HIGH"
        sig = "ALLEY_DRIFT_RIGHT"
    elif status == "ALLEY_CLEAR":
        target_text = "Hai bên là bức tường dài, phía trước thông thoáng, bạn hãy đi thẳng."
        target_prio = "LOW"
        sig = "ALLEY_CLEAR"
    else:  # CLEAR
        target_text = "Phía trước an toàn, đi thẳng."
        target_prio = "LOW"
        sig = "CLEAR"

    should_speak = False
    is_fresh_from_clear = (last_sig in ("CLEAR", "ALLEY_CLEAR") or not last_sig)

    if status == "STEPS":
        # Nếu vừa ở trạng thái CLEAR/ALLEY_CLEAR mà bước tới cầu thang/hố sâu -> Ngắt và cảnh báo NGAY LẬP TỨC (0.0s)!
        min_step_interval = 0.0 if is_fresh_from_clear else 1.2
        if (last_sig != sig and since >= min_step_interval) or since >= ALERT_REPEAT_INTERVAL_S:
            should_speak = True
    elif status in ("TURN_LEFT", "TURN_RIGHT", "TURN_AROUND", "ALLEY_DRIFT_LEFT", "ALLEY_DRIFT_RIGHT"):
        # 1. Nếu chuyển từ CLEAR / ALLEY_CLEAR sang có vật cản nguy hiểm hoặc lệch sát tường hẻm -> Cảnh báo TỨC THÌ (0.0s, frame 1)!
        # 2. Nếu người dùng vẫn tiếp tục tiến sát vào vật cản đó đến mức nguy kịch (<= 0.85m, giảm >= 0.45m so với lúc báo trước)
        #    -> Nhắc khẩn cấp sau 1.8s mà không cần chờ hết 6.0s!
        min_frames = 1 if is_fresh_from_clear else 2
        min_interval = 0.0 if is_fresh_from_clear else 1.2
        is_critical_approach = (min_dist <= 0.85 and (last_dist - min_dist) >= 0.45 and since >= 1.8)

        if (
            (last_sig != sig and st["blocked_frames"] >= min_frames and since >= min_interval)
            or is_critical_approach
            or since >= ALERT_REPEAT_INTERVAL_S
        ):
            should_speak = True
    elif status == "ALLEY_CLEAR":
        # Chế độ đi trong hẻm: Khi bắt đầu bước vào hẻm (last_sig != "ALLEY_CLEAR"), thông báo đúng 1 lần:
        # "Hai bên là bức tường dài, phía trước thông thoáng, bạn hãy đi thẳng."
        # Sau đó GIỮ IM LẶNG HOÀN TOÀN trong suốt quá trình đi thẳng trong hẻm, chỉ ngắt 0ms khi có vật cản phía trước!
        if last_sig != "ALLEY_CLEAR" and since >= (0.0 if not last_sig else 1.2):
            should_speak = True
    else:  # CLEAR
        # Giữ hướng an toàn >= 2 frame và sau khoảng nghỉ hợp lý
        if st["clear_frames"] >= 2 and since >= 1.5 and (last_sig != "CLEAR" or since >= CLEAR_REPEAT_INTERVAL_S):
            should_speak = True

    if should_speak:
        st["last_text"] = target_text
        st["last_priority"] = target_prio
        st["last_time"] = now
        st["last_alert_sig"] = sig
        st["last_dist_m"] = min_dist
        return target_text, target_prio, True

    return (st["last_text"] or target_text), st["last_priority"] if st["last_text"] else target_prio, False
