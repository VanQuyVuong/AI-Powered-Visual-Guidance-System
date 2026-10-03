import asyncio

class VoiceServiceAdapter:
    """
    Đây là Adapter để giao tiếp với API Blaze TTS của doanh nghiệp.
    Hiện tại vì chưa có API Key và tài liệu chính thức, chúng ta sử dụng Mock (Giả lập).
    Sau này có API thật, chỉ cần viết lại code trong hàm này là xong, hệ thống không bị ảnh hưởng.
    """
    
    @staticmethod
    async def generate_speech_url(text: str) -> str:
        # Giả lập thời gian chờ server doanh nghiệp xử lý giọng nói (1 giây)
        await asyncio.sleep(1)
        
        # Trả về một file âm thanh có thật trên mạng để App Flutter có thể test phát tiếng
        # Đây là link nhạc chuông mẫu, sau này sẽ là link do Blaze TTS trả về
        mock_audio_url = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
        
        print(f"[VoiceService] Đã giả lập tạo audio cho câu: '{text}'")
        return mock_audio_url
