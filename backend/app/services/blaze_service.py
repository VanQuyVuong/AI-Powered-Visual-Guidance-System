import os
import requests
from typing import Dict, Any, Optional

BLAZE_API_KEY = os.getenv("BLAZE_API_KEY", "68f5578def0d2c60d7cd49e9743f0c27ede87304")
BLAZE_TTS_URL = "https://api.blaze.vn/v1/tts"
BLAZE_STT_URL = "https://api.blaze.vn/v1/stt/execute"

class BlazeService:
    """
    Dịch vụ tích hợp giọng nói tiếng Việt từ Blaze.vn (Actable AI)
    Bao gồm Text-To-Speech (v1.5_pro) và Speech-To-Text (v1.0)
    """
    def __init__(self, api_key: str = BLAZE_API_KEY):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def text_to_speech(
        self,
        text: str,
        speaker_id: str = "HN-Nam-2-BL",
        audio_format: str = "mp3",
        audio_speed: str = "1",
        model: str = "v1.5_pro"
    ) -> Dict[str, Any]:
        """
        Chuyển văn bản cảnh báo/chỉ đường thành giọng nói tiếng Việt tự nhiên.
        Model: v1.5_pro
        """
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
        try:
            response = requests.post(
                BLAZE_TTS_URL,
                headers=self.headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            tts_id = data.get("id")
            audio_url = f"https://api.blaze.vn/v1/tts/{tts_id}/play" if tts_id else None
            return {
                "success": True,
                "id": tts_id,
                "audio_url": audio_url,
                "data": data
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

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
