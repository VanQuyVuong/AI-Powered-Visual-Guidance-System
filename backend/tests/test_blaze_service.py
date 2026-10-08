import unittest
from unittest.mock import patch, MagicMock
from app.services.blaze_service import BlazeService

class TestBlazeService(unittest.TestCase):
    def setUp(self):
        self.service = BlazeService(api_key="test_key_123")

    @patch("requests.post")
    def test_text_to_speech_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "job_abc123", "status": "pending"}
        mock_post.return_value = mock_response

        result = self.service.text_to_speech("Xin chào, có vật cản phía trước")
        self.assertTrue(result["success"])
        self.assertEqual(result["id"], "job_abc123")
        self.assertEqual(result["audio_url"], "http://api.blaze.vn/v1/tts/job_abc123/play")

    @patch("requests.post")
    def test_speech_to_text_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {
                "status_code": 200,
                "data": {
                    "transcription": "Dẫn đường đến chợ Bến Thành"
                }
            }
        }
        mock_post.return_value = mock_response

        result = self.service.speech_to_text(b"mock_audio_bytes")
        self.assertTrue(result["success"])
        self.assertEqual(result["transcription"], "Dẫn đường đến chợ Bến Thành")

if __name__ == "__main__":
    unittest.main()
