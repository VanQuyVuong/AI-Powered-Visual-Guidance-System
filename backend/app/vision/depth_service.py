"""
Depth Anything V2 Service – CPU-optimized
Load + inference chạy trên 1 worker thread (tránh kẹt event loop FastAPI trên Windows).
"""

import base64
import queue
import threading
import time

import cv2
import numpy as np

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_STRIDE = 3
INFER_TIMEOUT_SEC = 20

_model = None
_processor = None
_load_error = None
_state = "idle"  # idle | loading | ready | error
_frame_counter = 0
_last_depth_map = None
_job_q = queue.Queue(maxsize=1)
_worker_started = False
_start_lock = threading.Lock()


def _unavailable(status: str, error=None) -> dict:
    return {
        "depth_map": None,
        "depth_colormap_b64": None,
        "available": False,
        "cached": False,
        "error": error,
        "status": status,
    }


def _normalize_depth(raw: np.ndarray) -> np.ndarray:
    mn, mx = float(raw.min()), float(raw.max())
    if mx - mn < 1e-6:
        return np.zeros_like(raw, dtype=np.float32)
    return ((raw - mn) / (mx - mn)).astype(np.float32)


def _depth_to_colormap(depth_norm: np.ndarray) -> bytes:
    d8 = (depth_norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
    _, buf = cv2.imencode(".jpg", colored, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes()


def _load_model_sync():
    global _model, _processor, _load_error, _state
    _state = "loading"
    _load_error = None
    print("[Depth] Dang tai Depth Anything V2 Small...")
    t0 = time.time()
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        torch.set_grad_enabled(False)
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
        model.eval()
        _processor = processor
        _model = model
        _state = "ready"
        print(f"[Depth] Model san sang sau {time.time() - t0:.1f}s")
    except Exception as e:
        _model = None
        _processor = None
        _load_error = str(e)
        _state = "error"
        print(f"[Depth] Khong tai duoc model: {e}")
        raise


def _infer_sync(image_bytes: bytes) -> dict:
    global _frame_counter, _last_depth_map
    import torch
    from PIL import Image

    _frame_counter += 1
    use_cache = (_frame_counter % DEPTH_STRIDE != 0) and (_last_depth_map is not None)
    if use_cache:
        return {**_last_depth_map, "cached": True}

    nparr = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_cv is None:
        raise ValueError("Khong doc duoc anh")

    h, w = img_cv.shape[:2]
    max_dim = 384
    scale = min(1.0, max_dim / max(h, w))
    rw, rh = max(1, int(w * scale)), max(1, int(h * scale))
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img_rgb, (rw, rh), interpolation=cv2.INTER_AREA)
    pil_img = Image.fromarray(img_small)

    inputs = _processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        outputs = _model(**inputs)
        predicted = outputs.predicted_depth

    depth_raw = predicted.squeeze().cpu().numpy().astype(np.float32)
    if depth_raw.shape[:2] != (h, w):
        depth_raw = cv2.resize(depth_raw, (w, h), interpolation=cv2.INTER_LINEAR)

    depth_norm = _normalize_depth(depth_raw)
    b64 = base64.b64encode(_depth_to_colormap(depth_norm)).decode("utf-8")
    cache_result = {
        "depth_map": depth_norm,
        "depth_colormap_b64": b64,
        "available": True,
        "cached": False,
        "error": None,
        "status": "ready",
    }
    _last_depth_map = cache_result
    return cache_result


def _worker_loop():
    try:
        _load_model_sync()
    except Exception:
        return

    while True:
        item = _job_q.get()
        if item is None:
            break
        image_bytes, holder = item
        try:
            holder["result"] = _infer_sync(image_bytes)
        except Exception as e:
            print(f"[Depth] Loi inference: {e}")
            if _last_depth_map:
                holder["result"] = {**_last_depth_map, "cached": True}
            else:
                holder["result"] = _unavailable("error", str(e))
        finally:
            holder["event"].set()


def start_background_load():
    global _worker_started
    with _start_lock:
        if _worker_started:
            return
        _worker_started = True
        t = threading.Thread(target=_worker_loop, name="depth-worker", daemon=True)
        t.start()


def is_ready() -> bool:
    return _state == "ready" and _model is not None


def load_status() -> str:
    return _state


def public_status() -> dict:
    return {
        "state": _state,
        "ready": is_ready(),
        "error": _load_error,
        "cached": _last_depth_map is not None,
    }


def get_depth(image_bytes: bytes) -> dict:
    """
    Queue inference sang worker thread. Tra ve ngay neu model chua ready.
    """
    start_background_load()
    if _state != "ready":
        return _unavailable("error" if _load_error else "loading", _load_error)

    evt = threading.Event()
    holder = {"result": None, "event": evt}
    try:
        _job_q.put_nowait((image_bytes, holder))
    except queue.Full:
        if _last_depth_map:
            return {**_last_depth_map, "cached": True}
        return _unavailable("loading", "Depth dang ban, doi frame sau")

    if not evt.wait(timeout=INFER_TIMEOUT_SEC):
        if _last_depth_map:
            return {**_last_depth_map, "cached": True}
        return _unavailable("error", "Depth inference timeout")
    return holder["result"] or _unavailable("error", "Khong co ket qua depth")


def get_object_depth_value(depth_map: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    if depth_map is None:
        return -1.0
    H, W = depth_map.shape[:2]
    x1c = max(0, min(x1, W - 1))
    y1c = max(0, min(y1, H - 1))
    x2c = max(x1c + 1, min(x2, W))
    y2c = max(y1c + 1, min(y2, H))
    patch = depth_map[y1c:y2c, x1c:x2c]
    if patch.size == 0:
        return -1.0
    return float(np.median(patch))


def get_cached_depth() -> dict:
    if _last_depth_map is None:
        return _unavailable("loading")
    return {**_last_depth_map, "cached": True}
