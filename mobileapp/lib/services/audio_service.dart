import 'dart:convert';
import 'dart:typed_data';
import 'package:audioplayers/audioplayers.dart';

/// Service phát audio base64 nhận từ backend (Blaze TTS)
class AudioService {
  final AudioPlayer _player = AudioPlayer();
  bool _isPlaying = false;

  bool get isPlaying => _isPlaying;

  Future<void> init() async {
    _player.onPlayerStateChanged.listen((state) {
      _isPlaying = state == PlayerState.playing;
    });
  }

  /// Phát audio từ chuỗi base64 (backend trả về qua WebSocket)
  Future<void> playBase64Audio(String base64Audio) async {
    try {
      Uint8List audioBytes = base64Decode(base64Audio);
      await _player.stop();
      await _player.play(BytesSource(audioBytes));
    } catch (e) {
      print('[Audio] Lỗi phát audio: $e');
    }
  }

  Future<void> stop() async {
    await _player.stop();
  }

  void dispose() {
    _player.dispose();
  }
}
