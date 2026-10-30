import 'package:flutter_tts/flutter_tts.dart';
import '../config/app_config.dart';

/// Service Text-to-Speech — đọc cảnh báo bằng giọng tiếng Việt
class TtsService {
  final FlutterTts _flutterTts = FlutterTts();
  bool _isSpeaking = false;
  String _lastSpokenText = '';
  DateTime _lastSpeakTime = DateTime.now();

  bool get isSpeaking => _isSpeaking;

  Future<void> init() async {
    await _flutterTts.setLanguage("vi-VN");
    await _flutterTts.setSpeechRate(AppConfig.speechRate);
    await _flutterTts.setVolume(1.0);
    await _flutterTts.setPitch(1.0);

    _flutterTts.setStartHandler(() {
      _isSpeaking = true;
    });

    _flutterTts.setCompletionHandler(() {
      _isSpeaking = false;
    });

    _flutterTts.setCancelHandler(() {
      _isSpeaking = false;
    });

    _flutterTts.setErrorHandler((msg) {
      _isSpeaking = false;
      print('[TTS] Lỗi: $msg');
    });
  }

  /// Đọc text — tránh đọc trùng lặp trong 3 giây
  Future<void> speak(String text) async {
    if (text.trim().isEmpty) return;

    // Tránh đọc cùng 1 câu trong vòng 3 giây
    final now = DateTime.now();
    if (text == _lastSpokenText &&
        now.difference(_lastSpeakTime).inSeconds < 3) {
      return;
    }

    _lastSpokenText = text;
    _lastSpeakTime = now;

    // Nếu đang đọc câu cũ, dừng để đọc câu mới (ưu tiên cảnh báo mới nhất)
    if (_isSpeaking) {
      await _flutterTts.stop();
    }

    await _flutterTts.speak(text);
  }

  /// Đọc text ưu tiên cao (luôn dừng câu cũ, không check trùng)
  Future<void> speakUrgent(String text) async {
    if (text.trim().isEmpty) return;
    await _flutterTts.stop();
    _lastSpokenText = text;
    _lastSpeakTime = DateTime.now();
    await _flutterTts.speak(text);
  }

  /// Cập nhật tốc độ đọc
  Future<void> updateSpeechRate(double rate) async {
    await _flutterTts.setSpeechRate(rate);
  }

  Future<void> stop() async {
    await _flutterTts.stop();
    _isSpeaking = false;
  }
}
