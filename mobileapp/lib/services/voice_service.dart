import 'package:speech_to_text/speech_to_text.dart' as stt;

/// Service nhận diện giọng nói (Speech-to-Text) — điều khiển app bằng giọng
class VoiceService {
  final stt.SpeechToText _speech = stt.SpeechToText();
  bool _isInitialized = false;
  bool _isListening = false;

  bool get isInitialized => _isInitialized;
  bool get isListening => _isListening;

  /// Khởi tạo bộ nhận diện giọng nói
  Future<bool> init() async {
    try {
      _isInitialized = await _speech.initialize(
        onError: (val) => print('[STT] Lỗi: ${val.errorMsg}'),
        onStatus: (val) {
          print('[STT] Trạng thái: $val');
          if (val == 'done' || val == 'notListening') {
            _isListening = false;
          }
        },
      );
    } catch (e) {
      print('[STT] Lỗi khởi tạo: $e');
      _isInitialized = false;
    }
    return _isInitialized;
  }

  /// Lắng nghe giọng nói và trả về text
  void startListening(Function(String text) onResult) {
    if (!_isInitialized) {
      print('[STT] Chưa khởi tạo');
      return;
    }

    _isListening = true;
    _speech.listen(
      onResult: (val) {
        if (val.recognizedWords.isNotEmpty) {
          onResult(val.recognizedWords.toLowerCase());
        }
      },
      localeId: "vi_VN",
      listenMode: stt.ListenMode.confirmation,
      cancelOnError: true,
      partialResults: true,
    );
  }

  /// Dừng lắng nghe
  void stopListening() {
    _speech.stop();
    _isListening = false;
  }
}
