import 'package:speech_to_text/speech_to_text.dart' as stt;

class VoiceService {
  final stt.SpeechToText _speech = stt.SpeechToText();
  bool _isInitialized = false;

  /// Khởi tạo bộ nhận diện giọng nói
  Future<bool> init() async {
    _isInitialized = await _speech.initialize(
      onError: (val) => print('Lỗi STT: $val'),
      onStatus: (val) => print('Trạng thái STT: $val'),
    );
    return _isInitialized;
  }

  /// Lắng nghe giọng nói và trả về text liên tục
  void startListening(Function(String text) onResult) {
    if (!_isInitialized) return;
    
    _speech.listen(
      onResult: (val) {
        if (val.recognizedWords.isNotEmpty) {
          onResult(val.recognizedWords.toLowerCase());
        }
      },
      localeId: "vi_VN", // Ưu tiên nhận diện tiếng Việt
    );
  }

  /// Dừng lắng nghe
  void stopListening() {
    _speech.stop();
  }
}
