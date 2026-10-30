# backend/app/voice/api_integration.py
"""Voice API integration (TTS & STT) sử dụng Blaze.vn API.

Luồng TTS (async job):
  POST https://api.blaze.vn/v1/tts   body: {query, speaker_id, model, language, audio_format}
  → 202 + job id
  GET  https://api.blaze.vn/v1/tts/{id}/download  (hoặc /info → audio_url)

Tài liệu: https://blaze.vn/blog/tts-voice-cloning-eval-technical-report
"""
import os
import base64
import time
import threading
import requests
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env (không bắt buộc phải có python-dotenv)
# ---------------------------------------------------------------------------
def _manual_load_dotenv(path: Path) -> None:
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError as exc:
        print(f"⚠️  Không đọc được {path}: {exc}")


def _load_env() -> None:
    backend_dir = Path(__file__).resolve().parents[2]
    project_dir = backend_dir.parent
    candidates = [
        backend_dir / ".env",
        project_dir / ".env",
        Path.cwd() / ".env",
        Path.cwd() / "backend" / ".env",
    ]
    env_file = next((p for p in candidates if p.is_file()), None)
    if env_file is None:
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_file, override=False)
    except ImportError:
        _manual_load_dotenv(env_file)


_load_env()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TTS_ENDPOINT = os.getenv("BLAZE_TTS_URL", "https://api.blaze.vn/v1/tts")
STT_ENDPOINT = os.getenv("BLAZE_STT_URL", "https://api.blaze.vn/v1/stt")
VOICES_ENDPOINT = "https://api.blaze.vn/v1/voices"
API_KEY = os.getenv("BLAZE_API_KEY", "").strip()

VOICE_ID_ALERT = "HN-Nar-Nu-PhuongLy"
VOICE_ID_URGENT = "HN-Nar-Nam-BlazeNote"
VOICE_ID_CALM = "HN-Pod-Nu-NgocNhu"

TTS_MODEL = os.getenv("BLAZE_TTS_MODEL", "v2.0_flash")

if not API_KEY:
    print("[TTS] BLAZE_API_KEY chua duoc cau hinh trong backend/.env")
else:
    print(f"[TTS] Blaze API key loaded (...{API_KEY[-4:]})")


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def _pick_voice(priority: str) -> str:
    if priority in ("CRITICAL", "HIGH"):
        return VOICE_ID_URGENT
    if priority in ("MEDIUM", "LOW"):
        return VOICE_ID_ALERT
    return VOICE_ID_CALM


def _encode_audio(content: bytes) -> str:
    if not content or len(content) < 80:
        return ""
    return base64.b64encode(content).decode("utf-8")


def _download_audio(url: str) -> str:
    if url.startswith("/"):
        url = "https://api.blaze.vn" + url
    resp = requests.get(url, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=15)
    if resp.status_code == 200 and len(resp.content) > 80:
        return _encode_audio(resp.content)
    return ""


def _poll_tts_job(job_id: str) -> str:
    """Chờ job Blaze xong rồi tải audio (tối đa ~10s)."""
    download_url = f"{TTS_ENDPOINT.rstrip('/')}/{job_id}/download"
    info_url = f"{TTS_ENDPOINT.rstrip('/')}/{job_id}/info"

    for _ in range(8):
        time.sleep(0.7)
        try:
            dl = requests.get(download_url, headers=_headers(), timeout=10)
            ct = dl.headers.get("Content-Type", "")
            if dl.status_code == 200 and ("audio" in ct or len(dl.content) > 200):
                if "json" not in ct:
                    return _encode_audio(dl.content)
        except requests.RequestException:
            pass

        try:
            info = requests.get(info_url, headers=_headers(), timeout=8)
            if info.status_code != 200:
                continue
            if "audio" in info.headers.get("Content-Type", ""):
                return _encode_audio(info.content)
            if "json" not in info.headers.get("Content-Type", ""):
                continue
            data = info.json()
            status = str(data.get("status") or data.get("state") or "").lower()
            audio_b64 = data.get("audio_base64") or data.get("audio") or ""
            if audio_b64 and not audio_b64.startswith("http"):
                return audio_b64
            audio_url = data.get("audio_url") or data.get("url") or data.get("download_url") or ""
            if audio_url:
                encoded = _download_audio(audio_url)
                if encoded:
                    return encoded
            if status in ("failed", "error"):
                print(f"⚠️  Blaze TTS job failed: {data}")
                return ""
        except requests.RequestException:
            continue

    print("⚠️  Blaze TTS: hết thời gian chờ audio")
    return ""


def text_to_speech(text: str, priority: str = "MEDIUM") -> str:
    """Chuyển văn bản thành giọng nói (mp3 base64). Chuỗi rỗng nếu lỗi."""
    if not text or not text.strip():
        return ""
    if not API_KEY:
        print("⚠️  TTS bỏ qua: BLAZE_API_KEY chưa được cấu hình.")
        return ""

    payload = {
        "query": text.strip(),
        "speaker_id": _pick_voice(priority),
        "model": TTS_MODEL,
        "language": "vi",
        "audio_format": "mp3",
    }

    try:
        resp = requests.post(TTS_ENDPOINT, json=payload, headers=_headers(), timeout=15)

        if resp.status_code == 429:
            print("⚠️  Blaze TTS: Rate limit, thử lại sau 1s...")
            time.sleep(1)
            resp = requests.post(TTS_ENDPOINT, json=payload, headers=_headers(), timeout=15)

        if resp.status_code not in (200, 201, 202):
            print(f"⚠️  Blaze TTS API lỗi {resp.status_code}: {resp.text[:240]}")
            return ""

        content_type = resp.headers.get("Content-Type", "")
        if "audio" in content_type:
            return _encode_audio(resp.content)

        if "json" not in content_type:
            return _encode_audio(resp.content)

        data = resp.json()
        job_id = data.get("id") or data.get("job_id") or ""
        audio_b64 = data.get("audio_base64") or data.get("audio") or ""
        if audio_b64 and not str(audio_b64).startswith("http"):
            return audio_b64
        audio_url = data.get("audio_url") or data.get("url") or data.get("download_url") or ""
        if audio_url:
            encoded = _download_audio(audio_url)
            if encoded:
                return encoded
        if job_id:
            return _poll_tts_job(str(job_id))

        print(f"⚠️  Blaze TTS: JSON không có audio - keys: {list(data.keys())}")
        return ""

    except requests.Timeout:
        print("⚠️  Blaze TTS: timeout")
        return ""
    except requests.ConnectionError:
        print("⚠️  Blaze TTS: không kết nối được tới api.blaze.vn")
        return ""
    except Exception as e:
        print(f"⚠️  Blaze TTS lỗi không xác định: {e}")
        return ""


def speech_to_text(audio_path: str) -> str:
    file_path = Path(audio_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file audio: {audio_path}")

    with file_path.open("rb") as f:
        files = {"audio": f}
        resp = requests.post(
            STT_ENDPOINT,
            headers={"Authorization": f"Bearer {API_KEY}"},
            files=files,
            timeout=20,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"STT API lỗi {resp.status_code}: {resp.text}")
    result = resp.json()
    return result.get("transcript", "")


def get_available_voices() -> list:
    try:
        resp = requests.get(VOICES_ENDPOINT, headers=_headers(), timeout=8)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"⚠️  Không thể lấy danh sách voices: {e}")
    return []


def announce_detection(class_name: str) -> str:
    return text_to_speech(f"Phát hiện {class_name}", priority="MEDIUM")


# ---------------------------------------------------------------------------
# TTS nền: không chặn vòng WebSocket / YOLO
# ---------------------------------------------------------------------------
_tts_lock = threading.Lock()
_tts_busy = False
_pending_audio = ""
_last_spoken = ""
_last_spoken_time = 0.0
_queued_text = ""
_queued_priority = "MEDIUM"

# Cho phép lặp lại cùng câu hướng dẫn sau 10 giây
_REPEAT_COOLDOWN = 10.0


def queue_guidance_tts(text: str, priority: str = "MEDIUM") -> None:
    """Đưa câu hướng dẫn vào hàng đợi TTS (bỏ qua câu trùng / đang xử lý)."""
    global _tts_busy, _last_spoken, _last_spoken_time, _queued_text, _queued_priority
    if not text or not text.strip():
        return
    if (priority or "NONE").upper() == "NONE":
        return
    clean = text.strip()
    now = time.time()
    with _tts_lock:
        # Đang xử lý TTS → bỏ qua
        if _tts_busy:
            return
        # Câu trùng và chưa hết cooldown → bỏ qua
        if clean == _last_spoken and (now - _last_spoken_time) < _REPEAT_COOLDOWN:
            return
        _tts_busy = True
        _last_spoken = clean
        _last_spoken_time = now
        _queued_text = clean
        _queued_priority = priority

    threading.Thread(
        target=_tts_worker,
        args=(clean, priority),
        daemon=True,
    ).start()


def _tts_worker(text: str, priority: str) -> None:
    global _tts_busy, _pending_audio
    try:
        audio = text_to_speech(text, priority=priority)
        if audio:
            with _tts_lock:
                _pending_audio = audio
            print(f"[TTS] done audio_b64={len(audio)} text={text[:60]}")
        else:
            print(f"[TTS] no audio for: {text[:60]}")
    finally:
        with _tts_lock:
            _tts_busy = False


def pop_pending_tts() -> tuple:
    """Lấy audio vừa xong (nếu có). Trả về (audio_b64, text, priority)."""
    global _pending_audio
    with _tts_lock:
        audio = _pending_audio
        _pending_audio = ""
        return audio, _queued_text, _queued_priority
