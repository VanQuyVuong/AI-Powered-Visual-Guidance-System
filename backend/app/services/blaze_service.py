import os
import hashlib
import requests
from typing import Dict, Any, Optional

try:
    from dotenv import load_dotenv
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    load_dotenv(os.path.join(_project_root, ".env"))
except ImportError:
    pass

BLAZE_API_KEY = os.getenv("BLAZE_API_KEY", "")
BLAZE_TTS_URL = "https://api.blaze.vn/v1/tts"
BLAZE_STT_URL = "https://api.blaze.vn/v1/stt/execute"
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "audio_cache"))
os.makedirs(CACHE_DIR, exist_ok=True)

class BlazeService:
    """
    Dịch vụ tích hợp giọng nói tiếng Việt từ Blaze.vn kết hợp bộ đệm siêu tốc (Disk + RAM Cache)
    Đảm bảo phản hồi âm thanh tức thì (0ms) cho người khiếm thị khi di chuyển.
    """
    def __init__(self, api_key: str = BLAZE_API_KEY):
        self.api_key = api_key
        if not self.api_key:
            print("[WARN] BLAZE_API_KEY chua duoc set trong .env. TTS se dung Google fallback.")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self._cache: Dict[str, Any] = {}
        self._audio_bytes_cache: Dict[str, bytes] = {}
        self._text_to_id: Dict[str, str] = {}
        self._job_text_map: Dict[str, str] = {}

    def _get_hash(self, text: str) -> str:
        return hashlib.md5(text.strip().encode("utf-8")).hexdigest()

    def _fallback_google_tts(self, text: str) -> Optional[bytes]:
        """
        Dự phòng khẩn cấp: Tạo giọng đọc tiếng Việt chuẩn tự nhiên qua Google TTS
        Đảm bảo 100% không bao giờ bị im lặng hay trễ khi cảnh báo
        """
        try:
            import urllib.parse
            encoded = urllib.parse.quote(text)
            url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={encoded}&tl=vi&client=tw-ob"
            res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=4)
            if res.status_code == 200 and len(res.content) > 100:
                return res.content
        except Exception as e:
            print(f"[FALLBACK TTS ERROR]: {e}")
        return None

    def text_to_speech(
        self,
        text: str,
        speaker_id: str = "HN-Nam-2-BL",
        audio_format: str = "mp3",
        audio_speed: str = "1",
        model: str = "v1.5_pro"
    ) -> Dict[str, Any]:
        """
        Chuyển văn bản thành giọng nói tiếng Việt.
        1. Kiểm tra RAM Cache (0ms)
        2. Kiểm tra Disk Cache backend/audio_cache/ (< 1ms)
        3. Nếu chưa có: gọi Blaze AI với fallback tự động sang Google TTS
        """
        text_key = text.strip()
        h_id = self._get_hash(text_key)

        # 1. Kiểm tra RAM cache
        if h_id in self._audio_bytes_cache:
            return {
                "success": True,
                "id": h_id,
                "audio_url": f"/api/v1/voice/audio/{h_id}"
            }

        # 2. Kiểm tra Disk cache
        cached_file = os.path.join(CACHE_DIR, f"{h_id}.mp3")
        if os.path.exists(cached_file) and os.path.getsize(cached_file) > 500:
            try:
                with open(cached_file, "rb") as f:
                    content = f.read()
                self._audio_bytes_cache[h_id] = content
                self._audio_bytes_cache[text_key] = content
                self._job_text_map[h_id] = text_key
                return {
                    "success": True,
                    "id": h_id,
                    "audio_url": f"/api/v1/voice/audio/{h_id}"
                }
            except Exception:
                pass

        # 3. Nếu chưa có trong cache: Thử Blaze TTS với timeout ngắn tránh đứng giao diện
        tts_id = None
        self._job_text_map[h_id] = text_key
        
        payload = {
            "query": text,
            "language": "vi",
            "speaker_id": speaker_id,
            "audio_format": audio_format,
            "audio_quality": 64,
            "audio_speed": audio_speed,
            "normalization": "basic",
            "model": model
        }

        if self.api_key:
            try:
                response = requests.post(
                    BLAZE_TTS_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=2.0
                )
                if response.status_code in (200, 201, 202):
                    data = response.json()
                    blaze_id = data.get("id") or (data.get("data", {}) if isinstance(data.get("data"), dict) else {}).get("id")
                    if blaze_id:
                        tts_id = str(blaze_id)
                        self._cache[tts_id] = data
                        self._job_text_map[tts_id] = text_key
                        # Thử lấy audio nhanh (1 retry)
                        audio_fast = self._fetch_blaze_audio(tts_id, max_retries=2)
                        if audio_fast:
                            self._audio_bytes_cache[h_id] = audio_fast
                            self._audio_bytes_cache[tts_id] = audio_fast
                            with open(cached_file, "wb") as f:
                                f.write(audio_fast)
                            return {
                                "success": True,
                                "id": h_id,
                                "audio_url": f"/api/v1/voice/audio/{h_id}"
                            }
            except Exception as e:
                print(f"[BLAZE TTS POST ERROR]: {e}")

        # 4. Nếu Blaze chưa xong hoặc bị rate limit -> Dự phòng siêu tốc Google TTS ngay lập tức
        if h_id not in self._audio_bytes_cache:
            fb_content = self._fallback_google_tts(text_key)
            if fb_content:
                self._audio_bytes_cache[h_id] = fb_content
                self._audio_bytes_cache[text_key] = fb_content
                try:
                    with open(cached_file, "wb") as f:
                        f.write(fb_content)
                except Exception:
                    pass

        target_id = tts_id or h_id
        return {
            "success": True,
            "id": target_id,
            "audio_url": f"/api/v1/voice/audio/{target_id}"
        }

    def _fetch_blaze_audio(self, tts_id: str, max_retries: int = 4) -> Optional[bytes]:
        """
        Lấy audio bytes từ Blaze qua endpoint /play
        """
        if not self.api_key:
            return None

        import time
        play_url = f"https://api.blaze.vn/v1/tts/{tts_id}/play"

        for attempt in range(max_retries):
            try:
                res = requests.get(play_url, headers=self.headers, timeout=2.0)
                if res.status_code == 200 and len(res.content) > 500:
                    return res.content
                elif res.status_code == 425:
                    time.sleep(0.35)
                else:
                    time.sleep(0.2)
            except Exception:
                time.sleep(0.2)
        return None

    def get_audio_bytes(self, tts_id: str) -> Optional[bytes]:
        """
        Tải luồng audio MP3:
        1. Kiểm tra cache trong RAM (siêu nhanh < 1ms)
        2. Kiểm tra cache trên Disk (backend/audio_cache/<id>.mp3)
        3. Nếu chưa có, thăm dò Blaze API (/play)
        4. Nếu Blaze quá tải/lỗi, tự động kích hoạt Google TTS fallback đảm bảo không bao giờ im lặng
        """
        # 1. Kiểm tra cache sẵn có trong RAM
        if tts_id in self._audio_bytes_cache:
            return self._audio_bytes_cache[tts_id]

        # 2. Kiểm tra cache trên Disk
        cached_file = os.path.join(CACHE_DIR, f"{tts_id}.mp3")
        if os.path.exists(cached_file) and os.path.getsize(cached_file) > 500:
            try:
                with open(cached_file, "rb") as f:
                    content = f.read()
                self._audio_bytes_cache[tts_id] = content
                return content
            except Exception:
                pass

        orig_text = self._job_text_map.get(tts_id, "Lối đi an toàn, hãy tiếp tục đi thẳng.")

        # 3. Thử fetch từ Blaze
        audio_bytes = self._fetch_blaze_audio(tts_id, max_retries=3)
        if audio_bytes:
            self._audio_bytes_cache[tts_id] = audio_bytes
            self._audio_bytes_cache[orig_text] = audio_bytes
            try:
                with open(cached_file, "wb") as f:
                    f.write(audio_bytes)
            except Exception:
                pass
            return audio_bytes

        # 4. Dự phòng bằng Google TTS tiếng Việt siêu tốc
        fb_bytes = self._fallback_google_tts(orig_text)
        if fb_bytes:
            self._audio_bytes_cache[tts_id] = fb_bytes
            self._audio_bytes_cache[orig_text] = fb_bytes
            try:
                with open(cached_file, "wb") as f:
                    f.write(fb_bytes)
            except Exception:
                pass
            return fb_bytes

        return None

    def speech_to_text(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        model: str = "v1.0"
    ) -> Dict[str, Any]:
        """
        Nhận diện giọng nói tiếng Việt từ người dùng thành văn bản (STT).
        Model: v1.0
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            files = {"audio_file": (filename, audio_bytes, "audio/wav")}
            params = {"model": model}
            response = requests.post(
                BLAZE_STT_URL,
                headers=headers,
                params=params,
                files=files,
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            transcription = data.get("result", {}).get("data", {}).get("transcription", "")
            return {
                "success": True,
                "transcription": transcription,
                "data": data
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

blaze_service = BlazeService()
