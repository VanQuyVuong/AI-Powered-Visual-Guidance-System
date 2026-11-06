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
WALL_DANGER_M = 1.30        # Bức tường lại gần <= 1.3m mới bắt đầu cảnh báo rẽ/quay đầu (tránh báo từ xa 2m)
OBSTACLE_DANGER_M = 2.80    # Vật cản ở hành lang an toàn / làn giữa <= 2.8m (bắt trọn ghế, bàn, người phía trước)
SIDE_OBSTACLE_M = 1.80      # Vật cản ở 2 bên <= 1.8m

ALERT_REPEAT_INTERVAL_S = 6.0   # Nhắc lại sau 6s nếu vẫn đứng trước vật cản
CLEAR_REPEAT_INTERVAL_S = 10.0  # Nhắc lại "an toàn, đi thẳng" sau 10s

STEP_WORDS = ("bậc thềm", "bậc tam cấp", "cầu thang", "bậc thang")
WALL_CLASSES = {"wall"}


def _initial_state() -> dict:
    return {
        "phase": "PATROL",
        "last_text": "",
        "last_priority": "LOW",
        "last_time": 0.0,
        "last_alert_sig": "",
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
      - status: CLEAR | TURN_LEFT | TURN_RIGHT | TURN_AROUND | STEPS
      - name: tên vật cản (vật cản, bức tường, cột,...)
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

    # 1. BẬC THỀM / CẦU THANG
    is_steps = any(k in lower for k in STEP_WORDS)
    if is_steps:
        step_name = "cầu thang" if ("cầu thang" in lower or "bậc thang" in lower) else "bậc thềm"
        return {"status": "STEPS", "name": step_name}

    # 2. KIỂM TRA BỨC TƯỜNG (Chỉ nguy hiểm khi tiến sát <= WALL_DANGER_M)
    wall_objs = [
        o for o in objs
        if str(_get(o, "class_name", "")).lower() in WALL_CLASSES and dist(o) <= WALL_DANGER_M
    ]
    is_wall_near = bool(wall_objs) or (("bức tường" in lower) and (c_med <= WALL_DANGER_M))

    # 3. VẬT CẢN TRUNG TÂM / VÙNG HÀNH LANG AN TOÀN (CENTER / PATH)
    center_objs = [o for o in objs if is_in_path(o)]

    center_blocked = is_wall_near or bool(center_objs) or (c_blk and c_med <= 1.4)

    # 4. NẾU LÀN GIỮA THÔNG THOÁNG -> ĐI THẲNG AN TOÀN
    if not center_blocked:
        return {"status": "CLEAR", "name": "an toàn"}

    # 5. XÁC ĐỊNH TÊN VẬT CẢN CHẮN TRƯỚC MẶT (ƯU TIÊN TÊN VẬT THỂ YOLO THẬT SỰ)
    if len(center_objs) >= 2:
        name = "nhiều vật cản"
    elif center_objs:
        nearest = min(center_objs, key=dist)
        name = _get(nearest, "class_name_vn") or _get(nearest, "class_name") or "vật cản"
    elif is_wall_near:
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
        return {"status": "TURN_AROUND", "name": name}
    elif not left_blocked and right_blocked:
        return {"status": "TURN_LEFT", "name": name}
    elif left_blocked and not right_blocked:
        return {"status": "TURN_RIGHT", "name": name}
    else:
        # Cả hai bên đều còn khoảng trống: Chọn bên thoáng hơn
        if abs(left_gap - right_gap) >= 0.06:
            status = "TURN_LEFT" if left_gap > right_gap else "TURN_RIGHT"
        else:
            status = "TURN_LEFT" if l_med >= r_med else "TURN_RIGHT"
        return {"status": status, "name": name}


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

    st["clear_frames"] = st["clear_frames"] + 1 if status == "CLEAR" else 0
    st["blocked_frames"] = st["blocked_frames"] + 1 if status != "CLEAR" else 0

    since = (now - st["last_time"]) if st["last_time"] > 0 else float("inf")
    last_sig = st["last_alert_sig"]

    target_text = ""
    target_prio = "HIGH"
    sig = f"{status}:{name}"

    if status == "STEPS":
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
    else:  # CLEAR
        target_text = "Phía trước an toàn, đi thẳng."
        target_prio = "LOW"
        sig = "CLEAR"

    should_speak = False

    if status == "STEPS":
        if (last_sig != sig and since >= 1.5) or since >= ALERT_REPEAT_INTERVAL_S:
            should_speak = True
    elif status in ("TURN_LEFT", "TURN_RIGHT", "TURN_AROUND"):
        # Khi chuyển từ an toàn (CLEAR) sang có vật cản -> Báo ngay lập tức từ frame 1 để đảm bảo an toàn cho người khiếm thị!
        # Khi đang trong trạng thái cản mà chỉ đổi hướng rẽ -> cần 2 frame để chống rung nhấp nháy
        is_fresh_obstacle = (last_sig == "CLEAR" or not last_sig)
        min_frames = 1 if is_fresh_obstacle else 2
        min_interval = 0.8 if is_fresh_obstacle else 1.2
        if (last_sig != sig and st["blocked_frames"] >= min_frames and since >= min_interval) or since >= ALERT_REPEAT_INTERVAL_S:
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
        return target_text, target_prio, True

    return (st["last_text"] or target_text), st["last_priority"] if st["last_text"] else target_prio, False
