"""
CÔNG CỤ TEST ẢNH / VIDEO CHO MŨ THÔNG MINH
==========================================
Dùng đúng pipeline thật của backend (YOLO + ByteTrack + ROI + ước lượng khoảng cách + câu cảnh báo)
để kiểm tra AI nhận diện có đúng không. Kết quả được vẽ lên ảnh/video và lưu ra thư mục output.

CÁCH DÙNG (chạy từ thư mục gốc dự án):
  1) Thả ảnh/video của bạn vào thư mục:  test_media/input/
     rồi chạy:   python backend/test_media.py
  2) Hoặc chỉ định file cụ thể:
       python backend/test_media.py --file "D:/anh/duong_pho.jpg"
       python backend/test_media.py --file "D:/video/di_bo.mp4" --step 3
  3) Tham số:
       --file   đường dẫn ảnh/video (có thể lặp nhiều lần)
       --step   video: xử lý mỗi N frame (mặc định 3, nhỏ = chính xác hơn nhưng chậm hơn)
       --max-frames  video: giới hạn số frame xử lý (mặc định 0 = hết video)
       --expect "person,car"  (tuỳ chọn) danh sách vật thể bạn BIẾT chắc có trong ảnh/video,
                              công cụ sẽ báo ĐÚNG/THIẾU/THỪA so với kết quả AI.

Kết quả: test_media/output/<tên>_result.(jpg|mp4) + test_media/output/REPORT.md
"""
import argparse
import io
import os
import sys
import time
from collections import Counter, defaultdict

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPTS_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)

MEDIA_DIR = os.path.join(PROJECT_ROOT, "test_media")
INPUT_DIR = os.path.join(MEDIA_DIR, "input")
OUTPUT_DIR = os.path.join(MEDIA_DIR, "output")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Import pipeline thật của backend
from main import _analyze_frame_sync  # noqa: E402


def _load_font(size: int):
    for p in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\tahoma.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT_SM = _load_font(18)
FONT_LG = _load_font(24)


def encode_jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("Không mã hoá được frame")
    return buf.tobytes()


def draw_result(frame: np.ndarray, result: dict) -> np.ndarray:
    """Vẽ ROI, bounding box, nhãn tiếng Việt, khoảng cách và câu cảnh báo lên frame."""
    h, w = frame.shape[:2]
    out = frame.copy()

    # Vùng An Toàn ROI (hình thang) – cùng thông số với is_in_roi()
    top_y = int(h * 0.45)
    top_half = int(w * 0.35 / 2)
    bot_half = int(w * 0.75 / 2)
    cx = w // 2
    roi = np.array([[cx - top_half, top_y], [cx + top_half, top_y], [cx + bot_half, h - 1], [cx - bot_half, h - 1]], np.int32)
    overlay = out.copy()
    cv2.fillPoly(overlay, [roi], (0, 255, 120))
    out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)
    cv2.polylines(out, [roi], True, (0, 255, 120), 2)

    labels = []
    for obj in result.get("objects", []):
        b = obj["bounding_box"]
        danger = obj.get("is_danger")
        color = (60, 40, 255) if danger else ((0, 200, 255) if obj.get("in_roi") else (255, 200, 0))
        cv2.rectangle(out, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, 3 if danger else 2)
        tid = f"#{obj['id']} " if obj.get("id") is not None else ""
        text = f"{tid}{obj.get('class_name_vn') or obj['class_name']} {int(obj['confidence'] * 100)}% · {obj.get('distance_m', '?')}m"
        labels.append((b["x1"], max(0, b["y1"] - 26), text, color))

    # Vẽ chữ Unicode bằng PIL
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for x, y, text, color in labels:
        tw = d.textlength(text, font=FONT_SM)
        rgb = (color[2], color[1], color[0])
        d.rectangle([x, y, x + tw + 8, y + 24], fill=rgb)
        d.text((x + 4, y + 2), text, font=FONT_SM, fill=(255, 255, 255))

    g = result.get("guidance", {})
    gtext = g.get("text", "")
    prio = g.get("priority", "LOW")
    banner_rgb = (200, 30, 60) if prio == "HIGH" else (20, 120, 80)
    d.rectangle([0, h - 44, w, h], fill=banner_rgb)
    d.text((10, h - 38), f"[{prio}] {gtext}"[:110], font=FONT_LG, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def compare_expected(detected: Counter, expected: list):
    """So sánh với danh sách kỳ vọng. Trả về (đúng, thiếu, thừa)."""
    exp = {e.strip().lower() for e in expected if e.strip()}
    got = {k.lower() for k in detected}
    return sorted(exp & got), sorted(exp - got), sorted(got - exp)


def process_image(path: str, expected: list) -> dict:
    frame = cv2.imread(path)
    if frame is None:
        return {"name": os.path.basename(path), "type": "image", "error": "Không đọc được ảnh"}
    t0 = time.time()
    result = _analyze_frame_sync(encode_jpeg(frame))
    ms = (time.time() - t0) * 1000
    out = draw_result(frame, result)
    out_path = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(path))[0] + "_result.jpg")
    cv2.imwrite(out_path, out)

    classes = Counter(o["class_name"] for o in result["objects"])
    return {
        "name": os.path.basename(path), "type": "image", "out": out_path, "ms": ms,
        "classes": classes, "objects": result["objects"], "guidance": result["guidance"],
        "danger": sum(1 for o in result["objects"] if o.get("is_danger")),
        "expected": compare_expected(classes, expected) if expected else None,
    }


def process_video(path: str, step: int, max_frames: int, expected: list) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"name": os.path.basename(path), "type": "video", "error": "Không mở được video"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(path))[0] + "_result.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps / step), (w, h))

    classes = Counter()          # tổng số lần phát hiện theo lớp
    frames_with = defaultdict(int)  # số frame mà lớp đó xuất hiện
    latencies, danger_frames, guidance_counter = [], 0, Counter()
    processed, idx = 0, 0
    print(f"  Video: {w}x{h}, {fps:.1f}fps, {total} frame, xử lý mỗi {step} frame...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            t0 = time.time()
            result = _analyze_frame_sync(encode_jpeg(frame))
            latencies.append((time.time() - t0) * 1000)
            writer.write(draw_result(frame, result))
            names = {o["class_name"] for o in result["objects"]}
            for o in result["objects"]:
                classes[o["class_name"]] += 1
            for n in names:
                frames_with[n] += 1
            if any(o.get("is_danger") for o in result["objects"]):
                danger_frames += 1
            guidance_counter[result["guidance"]["text"]] += 1
            processed += 1
            if processed % 20 == 0:
                print(f"    ...{processed} frame đã xử lý ({idx}/{total})")
            if max_frames and processed >= max_frames:
                break
        idx += 1
    cap.release()
    writer.release()
    return {
        "name": os.path.basename(path), "type": "video", "out": out_path,
        "processed": processed, "total": total, "step": step,
        "avg_ms": float(np.mean(latencies)) if latencies else 0,
        "max_ms": float(np.max(latencies)) if latencies else 0,
        "fps_proc": 1000.0 / float(np.mean(latencies)) if latencies else 0,
        "classes": classes, "frames_with": dict(frames_with),
        "danger_frames": danger_frames, "guidance": guidance_counter.most_common(5),
        "expected": compare_expected(classes, expected) if expected else None,
    }


def write_report(results: list):
    lines = ["# 🎬 BÁO CÁO TEST ẢNH / VIDEO", f"*Thời gian:* {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    lines.append("> Mở file kết quả trong `test_media/output/` để **tự kiểm tra bằng mắt** AI nhận diện có đúng không.")
    lines.append("> Khung **đỏ** = vật cản nguy hiểm (trong làn đi & ≤1.8m) · **vàng** = trong làn nhưng còn xa · **xanh dương** = ngoài làn.\n")
    for r in results:
        lines.append(f"## {'🖼️' if r['type'] == 'image' else '🎞️'} {r['name']}")
        if r.get("error"):
            lines.append(f"- ❌ {r['error']}\n")
            continue
        lines.append(f"- **File kết quả:** `{os.path.relpath(r['out'], PROJECT_ROOT)}`")
        if r["type"] == "image":
            lines.append(f"- **Thời gian xử lý:** {r['ms']:.0f} ms · **Vật cản nguy hiểm:** {r['danger']}")
            lines.append(f"- **Câu cảnh báo:** {r['guidance']['text']} (`{r['guidance']['priority']}`)")
            lines.append("\n| Vật thể | Độ tin cậy | Vị trí | Khoảng cách | Trong làn | Nguy hiểm |\n|---|:-:|:-:|:-:|:-:|:-:|")
            for o in r["objects"]:
                lines.append(f"| {o.get('class_name_vn') or o['class_name']} ({o['class_name']}) | {o['confidence']:.2f} | {o['position']} | {o.get('distance_m')}m | {'✔' if o.get('in_roi') else '–'} | {'⚠️' if o.get('is_danger') else '–'} |")
        else:
            lines.append(f"- **Đã xử lý:** {r['processed']}/{r['total']} frame (mỗi {r['step']} frame)")
            lines.append(f"- **Tốc độ AI:** trung bình {r['avg_ms']:.0f} ms/frame (~{r['fps_proc']:.1f} FPS), chậm nhất {r['max_ms']:.0f} ms")
            lines.append(f"- **Frame có cảnh báo nguy hiểm:** {r['danger_frames']}/{r['processed']}")
            lines.append("\n| Vật thể | Số lần phát hiện | Xuất hiện ở số frame |\n|---|:-:|:-:|")
            for k, v in r["classes"].most_common():
                lines.append(f"| {k} | {v} | {r['frames_with'].get(k, 0)}/{r['processed']} |")
            if r["guidance"]:
                lines.append("\n**Câu cảnh báo hay gặp nhất:**")
                for t, c in r["guidance"]:
                    lines.append(f"- ({c} lần) {t}")
        if r.get("classes") and r["type"] == "image":
            lines.append(f"\n**Tổng hợp:** {dict(r['classes'])}")
        if r.get("expected"):
            ok_, miss, extra = r["expected"]
            lines.append(f"\n**So với kỳ vọng của bạn:** ✅ đúng {ok_} · ❌ AI bỏ sót {miss} · ⚠️ AI phát hiện thêm {extra}")
        lines.append("")
    path = os.path.join(OUTPUT_DIR, "REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main():
    ap = argparse.ArgumentParser(description="Test ảnh/video với AI Mũ Thông Minh")
    ap.add_argument("--file", action="append", default=[], help="đường dẫn ảnh/video (lặp được)")
    ap.add_argument("--step", type=int, default=3, help="video: xử lý mỗi N frame")
    ap.add_argument("--max-frames", type=int, default=0, help="video: giới hạn số frame xử lý (0 = hết)")
    ap.add_argument("--expect", default="", help='vật thể bạn biết chắc có, vd "person,car"')
    args = ap.parse_args()

    files = list(args.file)
    if not files:
        files = [os.path.join(INPUT_DIR, f) for f in sorted(os.listdir(INPUT_DIR))
                 if os.path.splitext(f)[1].lower() in IMAGE_EXT | VIDEO_EXT]
    if not files:
        print(f"⚠️ Chưa có file nào. Hãy thả ảnh/video vào:\n   {INPUT_DIR}\nhoặc dùng --file <đường dẫn>")
        return

    expected = [e for e in args.expect.split(",") if e.strip()]
    results = []
    for p in files:
        ext = os.path.splitext(p)[1].lower()
        print(f"\n▶ Đang test: {p}")
        if not os.path.exists(p):
            results.append({"name": p, "type": "image", "error": "File không tồn tại"})
            continue
        if ext in IMAGE_EXT:
            r = process_image(p, expected)
        elif ext in VIDEO_EXT:
            r = process_video(p, max(1, args.step), args.max_frames, expected)
        else:
            r = {"name": os.path.basename(p), "type": "image", "error": f"Định dạng {ext} không hỗ trợ"}
        results.append(r)
        if not r.get("error"):
            print(f"  ✅ Xong → {r['out']}")
            print(f"  Phát hiện: {dict(r['classes'])}")

    report = write_report(results)
    print(f"\n📄 Báo cáo: {report}")


if __name__ == "__main__":
    main()
