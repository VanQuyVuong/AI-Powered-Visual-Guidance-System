import 'package:flutter_tts/flutter_tts.dart';

class TtsService {
  final FlutterTts _flutterTts = FlutterTts();

  Future<void> init() async {
    await _flutterTts.setLanguage("vi-VN");
    await _flutterTts.setSpeechRate(0.5); // Tốc độ đọc vừa phải
  }

  Future<void> speak(String text) async {
    await _flutterTts.speak(text);
  }

  Future<void> stop() async {
    await _flutterTts.stop();
  }
}
