import 'dart:convert';
import 'package:web_socket_channel/web_socket_channel.dart';

class ApiService {
  // Đổi IP thành IPv4 của máy tính (VD: 192.168.1.100) nếu chạy trên máy ảo/thiết bị thật
  final String serverUrl = "ws://10.0.2.2:8000/api/v1/vision/stream";
  WebSocketChannel? _channel;

  /// Kết nối tới Backend qua WebSocket
  void connect({
    required Function(String warningText, String priority) onGuidanceReceived,
    required Function() onDone,
    required Function(dynamic error) onError,
  }) {
    _channel = WebSocketChannel.connect(Uri.parse(serverUrl));
    
    _channel!.stream.listen((message) {
      final data = jsonDecode(message);
      
      if (data['success'] == true) {
        // Backend đã làm phần AI và Decision Engine, Flutter chỉ việc lấy câu 'guidance' ra đọc
        final guidance = data['guidance'];
        if (guidance != null) {
          String text = guidance['text'];
          String priority = guidance['priority'];
          onGuidanceReceived(text, priority);
        }
      }
    }, onDone: onDone, onError: onError);
  }

  /// Nhận byte ảnh và nén thành chuỗi Base64 để gửi qua WebSocket
  void sendImage(List<int> imageBytes) {
    String base64Image = base64Encode(imageBytes);
    _channel?.sink.add(base64Image);
  }

  /// Đóng kết nối mạng
  void disconnect() {
    _channel?.sink.close();
    _channel = null;
  }
}
