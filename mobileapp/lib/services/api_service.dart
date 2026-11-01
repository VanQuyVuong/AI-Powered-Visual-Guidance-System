import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:http/http.dart' as http;
import '../config/app_config.dart';

/// Trạng thái kết nối WebSocket
enum ConnectionStatus { disconnected, connecting, connected, error }

/// Service quản lý kết nối tới Backend (WebSocket + REST)
class ApiService {
  WebSocketChannel? _channel;
  StreamSubscription? _subscription;
  Timer? _reconnectTimer;
  Timer? _pingTimer;

  ConnectionStatus _status = ConnectionStatus.disconnected;
  ConnectionStatus get status => _status;

  // Callbacks
  Function(String text, String priority, Map<String, dynamic> fullData)? _onGuidanceReceived;
  Function(String? audioBase64)? _onAudioReceived;
  Function(Map<String, dynamic> navigation)? _onNavigationReceived;
  Function(ConnectionStatus status)? _onStatusChanged;
  Function()? _onDone;
  Function(dynamic error)? _onError;

  int _reconnectAttempts = 0;
  static const int _maxReconnectAttempts = 10;
  bool _shouldReconnect = false;

  /// Cập nhật trạng thái và notify
  void _setStatus(ConnectionStatus newStatus) {
    _status = newStatus;
    _onStatusChanged?.call(newStatus);
  }

  /// Kết nối tới Backend qua WebSocket
  void connect({
    required Function(String text, String priority, Map<String, dynamic> fullData) onGuidanceReceived,
    Function(String? audioBase64)? onAudioReceived,
    Function(Map<String, dynamic> navigation)? onNavigationReceived,
    Function(ConnectionStatus status)? onStatusChanged,
    required Function() onDone,
    required Function(dynamic error) onError,
  }) {
    _onGuidanceReceived = onGuidanceReceived;
    _onAudioReceived = onAudioReceived;
    _onNavigationReceived = onNavigationReceived;
    _onStatusChanged = onStatusChanged;
    _onDone = onDone;
    _onError = onError;
    _shouldReconnect = true;
    _reconnectAttempts = 0;

    _connectInternal();
  }

  void _connectInternal() {
    _setStatus(ConnectionStatus.connecting);

    try {
      final wsUrl = AppConfig.wsUrl;
      print('[API] Đang kết nối tới: $wsUrl');

      _channel = WebSocketChannel.connect(Uri.parse(wsUrl));

      _subscription = _channel!.stream.listen(
        _handleMessage,
        onDone: () {
          print('[API] WebSocket đã đóng');
          _setStatus(ConnectionStatus.disconnected);
          if (_shouldReconnect) {
            _scheduleReconnect();
          }
          _onDone?.call();
        },
        onError: (error) {
          print('[API] Lỗi WebSocket: $error');
          _setStatus(ConnectionStatus.error);
          if (_shouldReconnect) {
            _scheduleReconnect();
          }
          _onError?.call(error);
        },
      );

      _setStatus(ConnectionStatus.connected);
      _reconnectAttempts = 0;

      // Ping mỗi 30 giây để giữ kết nối
      _pingTimer?.cancel();
      _pingTimer = Timer.periodic(const Duration(seconds: 30), (_) {
        try {
          _channel?.sink.add('ping');
        } catch (_) {}
      });
    } catch (e) {
      print('[API] Lỗi kết nối: $e');
      _setStatus(ConnectionStatus.error);
      if (_shouldReconnect) {
        _scheduleReconnect();
      }
    }
  }

  /// Xử lý message nhận từ backend
  void _handleMessage(dynamic message) {
    try {
      final data = jsonDecode(message);

      if (data['success'] == true) {
        // Xử lý guidance (cảnh báo)
        final guidance = data['guidance'];
        if (guidance != null) {
          String text = (guidance['text'] ?? '').toString();
          String priority = (guidance['priority'] ?? 'NONE').toString();

          // Gửi audio nếu có
          String? audioBase64 = guidance['audio_base64'];
          if (audioBase64 != null && audioBase64.isNotEmpty) {
            _onAudioReceived?.call(audioBase64);
          }

          // Chỉ callback khi có text thật
          if (text.trim().isNotEmpty) {
            _onGuidanceReceived?.call(text, priority, data);
          }
        }

        // Xử lý navigation direction
        final navigation = data['navigation'];
        if (navigation != null) {
          _onNavigationReceived?.call(Map<String, dynamic>.from(navigation));
        }
      }
    } catch (e) {
      print('[API] Lỗi parse JSON: $e');
    }
  }

  /// Tự động reconnect với exponential backoff
  void _scheduleReconnect() {
    if (_reconnectAttempts >= _maxReconnectAttempts) {
      print('[API] Đã thử $maxReconnectAttempts lần, dừng reconnect');
      _setStatus(ConnectionStatus.error);
      return;
    }

    _reconnectAttempts++;
    final delay = Duration(seconds: _reconnectAttempts * 2); // 2s, 4s, 6s, ...
    print('[API] Reconnect lần $_reconnectAttempts sau ${delay.inSeconds}s...');

    _reconnectTimer?.cancel();
    _reconnectTimer = Timer(delay, () {
      if (_shouldReconnect) {
        _connectInternal();
      }
    });
  }

  /// Gửi ảnh (bytes) qua WebSocket dưới dạng Base64
  void sendImage(List<int> imageBytes) {
    if (_status != ConnectionStatus.connected) return;
    try {
      String base64Image = base64Encode(imageBytes);
      _channel?.sink.add(base64Image);
    } catch (e) {
      print('[API] Lỗi gửi ảnh: $e');
    }
  }

  /// Đóng kết nối
  void disconnect() {
    _shouldReconnect = false;
    _reconnectTimer?.cancel();
    _pingTimer?.cancel();
    _subscription?.cancel();
    _channel?.sink.close();
    _channel = null;
    _setStatus(ConnectionStatus.disconnected);
  }

  /// Gửi ảnh lên API OCR để đọc chữ
  Future<String?> readTextFromImage(List<int> imageBytes) async {
    final restUrl = '${AppConfig.httpBaseUrl}/api/v1/vision/read_text';

    try {
      var request = http.MultipartRequest('POST', Uri.parse(restUrl));
      request.files.add(http.MultipartFile.fromBytes(
        'image',
        imageBytes,
        filename: 'ocr_frame.jpg',
      ));

      var response = await request.send().timeout(const Duration(seconds: 15));
      if (response.statusCode == 200) {
        String responseBody = await response.stream.bytesToString();
        var data = jsonDecode(responseBody);
        if (data['success'] == true) {
          return data['text'];
        }
      }
    } catch (e) {
      print('[API] Lỗi OCR: $e');
    }
    return null;
  }

  /// Chuyển chế độ NAVIGATION / CONVERSATION
  Future<String?> setMode(String mode) async {
    final url = '${AppConfig.httpBaseUrl}/api/v1/vision/mode';
    try {
      final response = await http.post(
        Uri.parse(url),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'mode': mode}),
      ).timeout(const Duration(seconds: 5));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['message'];
      }
    } catch (e) {
      print('[API] Lỗi set mode: $e');
    }
    return null;
  }

  /// Gửi câu hỏi cho AI (chế độ hỏi đáp)
  Future<String?> askQuestion(String question, List<Map<String, dynamic>> detections) async {
    final url = '${AppConfig.httpBaseUrl}/api/v1/vision/qa';
    try {
      final response = await http.post(
        Uri.parse(url),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'question': question,
          'detections': detections,
        }),
      ).timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['answer'];
      }
    } catch (e) {
      print('[API] Lỗi Q&A: $e');
    }
    return null;
  }

  /// Kiểm tra backend có hoạt động không
  Future<bool> testConnection() async {
    try {
      final response = await http
          .get(Uri.parse('${AppConfig.httpBaseUrl}/api/v1/vision/depth/status'))
          .timeout(const Duration(seconds: 5));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  static int get maxReconnectAttempts => _maxReconnectAttempts;
}
