import 'package:shared_preferences/shared_preferences.dart';

/// Lớp quản lý cấu hình app — lưu trữ IP server, tốc độ giọng nói, v.v.
class AppConfig {
  static const String _keyServerIp = 'server_ip';
  static const String _keyServerPort = 'server_port';
  static const String _keySpeechRate = 'speech_rate';
  static const String _keyVibrationEnabled = 'vibration_enabled';
  static const String _keyScanInterval = 'scan_interval_ms';

  // Giá trị mặc định
  static const String defaultIp = '192.168.1.67';
  static const String defaultPort = '8000';
  static const double defaultSpeechRate = 0.5;
  static const bool defaultVibrationEnabled = true;
  static const int defaultScanInterval = 1500;

  static SharedPreferences? _prefs;

  static Future<void> init() async {
    _prefs = await SharedPreferences.getInstance();
  }

  // ── Server IP ──
  static String get serverIp =>
      _prefs?.getString(_keyServerIp) ?? defaultIp;

  static set serverIp(String value) =>
      _prefs?.setString(_keyServerIp, value);

  // ── Server Port ──
  static String get serverPort =>
      _prefs?.getString(_keyServerPort) ?? defaultPort;

  static set serverPort(String value) =>
      _prefs?.setString(_keyServerPort, value);

  // ── WebSocket URL (tự tạo từ IP + Port) ──
  static String get wsUrl =>
      'ws://$serverIp:$serverPort/api/v1/vision/stream';

  static String get httpBaseUrl =>
      'http://$serverIp:$serverPort';

  // ── Tốc độ giọng đọc ──
  static double get speechRate =>
      _prefs?.getDouble(_keySpeechRate) ?? defaultSpeechRate;

  static set speechRate(double value) =>
      _prefs?.setDouble(_keySpeechRate, value);

  // ── Rung cảnh báo ──
  static bool get vibrationEnabled =>
      _prefs?.getBool(_keyVibrationEnabled) ?? defaultVibrationEnabled;

  static set vibrationEnabled(bool value) =>
      _prefs?.setBool(_keyVibrationEnabled, value);

  // ── Thời gian giữa các lần quét (ms) ──
  static int get scanInterval =>
      _prefs?.getInt(_keyScanInterval) ?? defaultScanInterval;

  static set scanInterval(int value) =>
      _prefs?.setInt(_keyScanInterval, value);
}
