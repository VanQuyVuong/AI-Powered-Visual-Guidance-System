"""
Công cụ đánh giá tổng hợp 4 mô hình AI & hiệu chuẩn khoảng cách trên thư mục `ImagesTest/`.

Cách dùng:
  .venv\\Scripts\\python.exe backend/evaluate_images_folder.py
  .venv\\Scripts\\python.exe backend/evaluate_images_folder.py --dir ImagesTest --write-env
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CALIBRATION_NAME_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*m(?=\b|_|\.)", re.IGNORECASE)

EXPECTED_RULES = {
    "cau_thang": {"label": "Cầu thang / Bậc thềm", "expect_any": {"stairs", "step", "curb", "doorstep"}},
    "ho_sau": {"label": "Hố sâu / Ổ gà", "expect_any": {"dropoff", "pothole", "open_manhole"}},
    "nap_cong": {"label": "Nắp cống / Hố ga", "expect_any": {"open_manhole", "pothole", "dropoff"}},
    "guong_kinh": {"label": "Gương / Cửa kính", "expect_any": {"mirror", "glass_door", "door"}},
    "tuong_cot": {
        "label": "Tường / Cột / Rào chắn / Vật cản",
        "expect_any": {"wall", "column", "pillar", "obstacle", "door", "cabinet", "bench", "chair"},
    },
    "xe_co_nguoi": {
        "label": "Người / Xe cộ / Ghế / Động vật",
        "expect_any": {"person", "motorcycle", "bicycle", "car", "bus", "truck", "bench", "chair", "dog", "obstacle", "column"},
    },
    "nen_gach": {
        "label": "Nền gạch / Lối đi trống",
        "forbid_any": {"stairs", "step", "wall", "dropoff", "pothole", "open_manhole", "mirror"},
    },
    "vat_can_xa": {"label": "Vật cản xa (> 2.4m)", "expect_no_danger_if_far": True},
}


def parse_true_distance_m(filename: str) -> Optional[float]:
    """Đọc khoảng cách thật gắn trong tên file (VD: `ghe_2.8m.jpg` -> 2.8)."""
    match = CALIBRATION_NAME_RE.search(filename)
    if not match:
        return None
    try:
        val = float(match.group(1).replace(",", "."))
        return val if 0.2 <= val <= 15.0 else None
    except ValueError:
        return None


def wait_for_models(timeout_s: float = 45.0) -> None:
    """Đợi Depth Anything V2, SegFormer và EdgeSAM nạp xong hoàn toàn trước khi chấm ảnh."""
    from app.vision.depth_service import depth_service
    from app.vision.sam_service import sam_service

    t0 = time.time()
    while not getattr(depth_service, "load_complete", False) and (time.time() - t0) < timeout_s:
        if depth_service.is_loaded and depth_service.seg_loaded and depth_service.pothole_model is not None:
            break
        time.sleep(0.2)
    sam_service._ensure_loaded()


def evaluate_category(category: str, result: Dict[str, Any], true_dist_m: Optional[float]) -> tuple[str, str]:
    """Đánh giá ĐẠT / CẦN KIỂM TRA theo thư mục kịch bản."""
    objs = result.get("objects") or []
    classes = {str(o.get("class_name", "")).lower() for o in objs}
    state_code = (
        result.get("navigation", {})
        .get("fusion_summary", {})
        .get("decision_telemetry", {})
        .get("state_code", "CLEAR")
    )

    if category in ("cau_thang", "ho_sau", "nap_cong", "guong_kinh", "tuong_cot", "xe_co_nguoi"):
        expected = EXPECTED_RULES[category]["expect_any"]
        if classes & expected:
            return "PASS", f"Bắt trúng {sorted(classes & expected)}"
        return "FAIL", f"Chưa bắt được nhóm {sorted(expected)} (hiện có: {sorted(classes) or ['none']})"

    if category == "nen_gach":
        forbidden = EXPECTED_RULES["nen_gach"]["forbid_any"]
        bad = classes & forbidden
        if not bad and state_code in ("CLEAR", "ALLEY_CLEAR"):
            return "PASS", "Nền gạch sạch, đi thẳng an toàn"
        return "FAIL", f"Báo giả trên nền gạch: classes={sorted(bad)}, state={state_code}"

    if category == "vat_can_xa":
        danger_objs = [o for o in objs if o.get("is_danger")]
        if true_dist_m is not None and true_dist_m >= 2.4:
            if not danger_objs:
                return "PASS", f"Vật cách {true_dist_m}m -> Không báo nguy hiểm giả"
            return "FAIL", f"Vật thực tế {true_dist_m}m nhưng vẫn đánh nguy hiểm: {[(o.get('class_name'), o.get('distance_m')) for o in danger_objs]}"
        if not danger_objs:
            return "PASS", "Không có cảnh báo nguy hiểm giả cho vật ở xa"
        return "WARN", f"Có vật bị đánh nguy hiểm ở {[(o.get('class_name'), o.get('distance_m')) for o in danger_objs]}"

    return "INFO", f"Trạng thái: {state_code}"


def render_visual_result(
    frame_bgr: np.ndarray,
    result: Dict[str, Any],
    out_path: Path,
    category: str,
    verdict: str,
) -> None:
    """Vẽ ảnh trực quan gồm bounding box, nhãn tiếng Việt, khoảng cách mét và depth heatmap bên cạnh."""
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]

    for obj in result.get("objects") or []:
        b = obj.get("bounding_box") or {}
        x1, y1, x2, y2 = int(b.get("x1", 0)), int(b.get("y1", 0)), int(b.get("x2", 0)), int(b.get("y2", 0))
        is_danger = bool(obj.get("is_danger"))
        color = (60, 45, 255) if is_danger else (255, 220, 40)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        lbl = f"{obj.get('class_name', '')} {obj.get('distance_m', '--')}m"
        cv2.putText(vis, lbl, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    banner = result.get("guidance", {}).get("display_text") or result.get("guidance", {}).get("text") or ""
    header_txt = f"[{category} | {verdict}] {result.get('processing_ms', 0)}ms"
    cv2.rectangle(vis, (0, 0), (w, 52), (18, 22, 32), -1)
    cv2.putText(vis, header_txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 160) if verdict == "PASS" else (0, 200, 255), 2)
    cv2.putText(vis, banner[:70], (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 244, 252), 1)

    # Ghép thêm colormap depth bên phải nếu có
    import base64
    depth_b64 = result.get("depth_b64")
    if depth_b64:
        try:
            d_bytes = base64.b64decode(depth_b64)
            d_img = cv2.imdecode(np.frombuffer(d_bytes, np.uint8), cv2.IMREAD_COLOR)
            if d_img is not None:
                d_resized = cv2.resize(d_img, (w, h))
                vis = np.hstack([vis, d_resized])
        except Exception:
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def update_env_depth_scale(scale: float) -> None:
    """Cập nhật hệ số DEPTH_SCALE_FACTOR vào file .env ở thư mục gốc dự án."""
    env_path = PROJECT_ROOT / ".env"
    lines: List[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    updated = False
    new_lines = []
    for line in lines:
        if line.strip().startswith("DEPTH_SCALE_FACTOR="):
            new_lines.append(f"DEPTH_SCALE_FACTOR={scale:.3f}")
            updated = True
        else:
            new_lines.append(line)
    if not updated:
        new_lines.append(f"DEPTH_SCALE_FACTOR={scale:.3f}")
    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Đánh giá & hiệu chuẩn 4 mô hình AI trên thư mục ImagesTest.")
    parser.add_argument("--dir", default=str(PROJECT_ROOT / "ImagesTest"), help="Đường dẫn thư mục ảnh kiểm thử.")
    parser.add_argument("--write-env", action="store_true", help="Tự động ghi DEPTH_SCALE_FACTOR vào .env nếu có ảnh mẫu gắn mét.")
    args = parser.parse_args()

    target_dir = Path(args.dir)
    if not target_dir.exists():
        print(f"❌ Không tìm thấy thư mục ảnh: {target_dir}")
        return 1

    images = sorted(
        p for p in target_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and "_results" not in p.parts
    )
    if not images:
        print(f"📂 Thư mục `{target_dir}` hiện chưa có ảnh nào (.jpg/.png/.webp).")
        print("👉 Hãy thả ảnh cầu thang, nền gạch, hố sâu, vật cản xa vào `ImagesTest/` (hoặc các thư mục con) rồi chạy lại lệnh này!")
        return 0

    from app.vision.depth_service import depth_service
    from app.vision.guided_scan import reset_voice_state
    from app.vision.pipeline import analyze_frame_sync

    print("⏳ Đang nạp cả 4 mô hình AI (YOLO11 + Depth V2 + EdgeSAM + SegFormer)...")
    wait_for_models()
    print(f"✅ Đã sẵn sàng! Tìm thấy {len(images)} ảnh cần đánh giá.\n")

    out_dir = target_dir / "_results"
    summary_records: List[Dict[str, Any]] = []
    scale_ratios: List[float] = []
    counts = {"PASS": 0, "FAIL": 0, "WARN": 0, "INFO": 0}

    for idx, img_path in enumerate(images, 1):
        rel = img_path.relative_to(target_dir)
        category = rel.parts[0] if len(rel.parts) > 1 else "root"
        img_bytes = img_path.read_bytes()
        frame_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            print(f"  [{idx}/{len(images)}] ⚠️ Bỏ qua file lỗi đọc ảnh: {rel}")
            continue

        # Reset state và ép Depth Map + SegFormer + Pothole chạy đủ cho từng ảnh tĩnh
        reset_voice_state()
        depth_service.compute_depth_map(frame_bgr, force_aux=True)
        result = analyze_frame_sync(img_bytes, is_stream=False, include_visuals=True)

        true_dist = parse_true_distance_m(img_path.name)
        verdict, detail = evaluate_category(category, result, true_dist)
        counts[verdict] = counts.get(verdict, 0) + 1

        objs_brief = [
            f"{o.get('class_name')}({o.get('distance_m')}m{'!' if o.get('is_danger') else ''})"
            for o in (result.get("objects") or [])
        ]
        print(f"[{idx:02d}/{len(images):02d}] {str(rel):32s} -> {verdict:4s} | {result.get('processing_ms'):5.1f}ms | {objs_brief or ['CLEAR']} | {detail}")

        if true_dist is not None and result.get("objects"):
            center_objs = [o for o in result["objects"] if o.get("position") == "center"] or result["objects"]
            pred_d = float(center_objs[0].get("distance_m") or 0.0)
            if pred_d > 0.2:
                ratio = true_dist / pred_d
                scale_ratios.append(ratio)
                print(f"       ↳ Đối chiếu khoảng cách thật={true_dist:.2f}m vs đo={pred_d:.2f}m (tỷ lệ {ratio:.2f}x)")

        render_visual_result(frame_bgr, result, out_dir / rel.with_suffix(".jpg"), category, verdict)
        summary_records.append({
            "file": str(rel),
            "category": category,
            "verdict": verdict,
            "detail": detail,
            "true_distance_m": true_dist,
            "processing_ms": result.get("processing_ms"),
            "objects": result.get("objects"),
            "guidance": result.get("guidance"),
        })

    report_path = out_dir / "report.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary_records, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"📊 TỔNG KẾT: PASS={counts['PASS']} | FAIL={counts['FAIL']} | WARN={counts['WARN']} | INFO={counts['INFO']}")
    print(f"🖼️  Ảnh trực quan + báo cáo chi tiết đã lưu tại: {out_dir}")

    if scale_ratios:
        rec_scale = float(np.median(scale_ratios))
        print(f"📏 Hệ số hiệu chuẩn DEPTH_SCALE_FACTOR đề xuất cho camera của bạn: {rec_scale:.3f}")
        if args.write_env:
            update_env_depth_scale(rec_scale)
            print(f"💾 Đã ghi DEPTH_SCALE_FACTOR={rec_scale:.3f} vào {PROJECT_ROOT / '.env'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
