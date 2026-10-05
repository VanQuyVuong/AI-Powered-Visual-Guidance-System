import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:flutter_tts/flutter_tts.dart';

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    cameras = await availableCameras();
  } on CameraException catch (e) {
    print('Error in fetching the cameras: $e');
  }
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Visual Guidance',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
      ),
      home: const VisionScreen(),
    );
  }
}

class VisionScreen extends StatefulWidget {
  const VisionScreen({super.key});
  @override
  State<VisionScreen> createState() => _VisionScreenState();
}

class _VisionScreenState extends State<VisionScreen> {
  CameraController? _controller;
  WebSocketChannel? _channel;
  FlutterTts flutterTts = FlutterTts();
  
  bool _isStreaming = false;
  bool _isProcessingFrame = false;
  String _latestWarning = "Bấm Bắt đầu để quét";
  
  // Sửa lại IP này thành IPv4 của máy tính (VD: 192.168.1.100) nếu chạy trên máy ảo/điện thoại thật
  final String serverUrl = "ws://10.0.2.2:8000/api/v1/vision/stream";

  @override
  void initState() {
    super.initState();
    _initCamera();
    _initTts();
  }

  Future<void> _initCamera() async {
    if (cameras.isEmpty) return;
    _controller = CameraController(
      cameras[0], // Camera sau
      ResolutionPreset.low, // Gửi ảnh độ phân giải thấp cho lẹ
      enableAudio: false,
    );
    await _controller!.initialize();
    if (mounted) setState(() {});
  }

  Future<void> _initTts() async {
    await flutterTts.setLanguage("vi-VN");
    await flutterTts.setSpeechRate(0.5);
  }

  void _startStreaming() {
    if (_controller == null || !_controller!.value.isInitialized) return;
    
    // Mở kết nối tới FastAPI WebSocket
    _channel = WebSocketChannel.connect(Uri.parse(serverUrl));
    
    setState(() {
      _isStreaming = true;
      _latestWarning = "Đã kết nối Server. Đang quét...";
    });
    
    // Lắng nghe phản hồi từ Server
    _channel!.stream.listen((message) {
      final data = jsonDecode(message);
      if (data['success'] == true) {
        List objects = data['objects'] ?? [];
        if (objects.isNotEmpty) {
          // Lấy tên vật thể và cảnh báo
          List<String> warnings = objects.map((obj) => obj['class'].toString()).toList();
          String warningText = "Cẩn thận, có ${warnings.join(', ')} phía trước.";
          
          setState(() {
            _latestWarning = warningText;
          });
          
          flutterTts.speak(warningText);
        }
      }
    }, onDone: () {
      _stopStreaming();
    }, onError: (error) {
      print("Lỗi WebSocket: $error");
      _stopStreaming();
    });

    // Bắt đầu chụp ảnh liên tục từ Camera
    _controller!.startImageStream((CameraImage image) async {
      if (_isProcessingFrame || !_isStreaming) return;
      _isProcessingFrame = true;
      
      try {
        // Ghi chú: Chuyển đổi YUV420 sang JPEG trong Flutter khá phức tạp.
        // Ở phiên bản MVP này, ta chỉ gửi tín hiệu Ping để test luồng dữ liệu trước
        // (Trong thực tế cần hàm chuyển đổi YUV -> JPEG để gửi qua base64)
        _channel?.sink.add("Ping from Flutter"); 
      } catch (e) {
        print(e);
      }
      
      await Future.delayed(const Duration(milliseconds: 500)); // Gửi 2 frame/giây
      _isProcessingFrame = false;
    });
  }

  void _stopStreaming() {
    setState(() {
      _isStreaming = false;
      _latestWarning = "Đã dừng quét.";
    });
    _controller?.stopImageStream();
    _channel?.sink.close();
  }

  @override
  void dispose() {
    _controller?.dispose();
    _channel?.sink.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text("Kính Thông Minh AI"),
        backgroundColor: Colors.teal,
      ),
      body: Column(
        children: [
          Expanded(
            child: _controller == null || !_controller!.value.isInitialized
                ? const Center(child: CircularProgressIndicator())
                : CameraPreview(_controller!),
          ),
          Container(
            padding: const EdgeInsets.all(20),
            color: Colors.white,
            width: double.infinity,
            child: Column(
              children: [
                Text(
                  _latestWarning,
                  style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: Colors.red),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 20),
                ElevatedButton.icon(
                  onPressed: _isStreaming ? _stopStreaming : _startStreaming,
                  icon: Icon(_isStreaming ? Icons.stop : Icons.play_arrow),
                  label: Text(_isStreaming ? "DỪNG QUÉT" : "BẮT ĐẦU QUÉT"),
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 15),
                    backgroundColor: _isStreaming ? Colors.red : Colors.teal,
                    foregroundColor: Colors.white,
                  ),
                )
              ],
            ),
          )
        ],
      ),
    );
  }
}
