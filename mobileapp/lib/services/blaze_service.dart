import 'dart:convert';
import 'package:http/http.dart' as http;

class BlazeService {
  static const String apiKey = "68f5578def0d2c60d7cd49e9743f0c27ede87304";
  static const String ttsUrl = "https://api.blaze.vn/v1/tts";
  static const String sttUrl = "https://api.blaze.vn/v1/stt/execute?model=v1.0";

  /// Chuyển văn bản thành audio url sử dụng Blaze TTS (v1.5_pro)
  static Future<String?> textToSpeech(
    String text, {
    String speakerId = "HN-Nam-2-BL",
    String audioSpeed = "1",
  }) async {
    try {
      final response = await http.post(
        Uri.parse(ttsUrl),
        headers: {
          "Authorization": "Bearer $apiKey",
          "Content-Type": "application/json",
        },
        body: jsonEncode({
          "query": text,
          "language": "vi",
          "speaker_id": speakerId,
          "audio_format": "mp3",
          "audio_quality": 64,
          "audio_speed": audioSpeed,
          "normalization": "basic",
          "model": "v1.5_pro",
        }),
      );

      if (response.statusCode == 200 || response.statusCode == 201) {
        final data = jsonDecode(response.body);
        final String? id = data['id'];
        if (id != null) {
          return "http://api.blaze.vn/v1/tts/$id/play";
        }
      } else {
        print("Blaze TTS HTTP Error: ${response.statusCode} - ${response.body}");
      }
    } catch (e) {
      print("Lỗi kết nối Blaze TTS: $e");
    }
    return null;
  }
}
