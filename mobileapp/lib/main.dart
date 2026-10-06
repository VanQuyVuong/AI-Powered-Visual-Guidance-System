import 'dart:async';
import 'dart:convert';
import 'dart:io';
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
  Timer? _timer;
  
  bool _isStreaming = false;
  bool _isProcessingFrame = false;
  String _latestWarning = "Bấm Bắt đầu để quét";
  
  // IP của Server. Nếu chạy bằng máy thật và cắm cáp, hãy nhập IP LAN của máy tính (vd: 192.168.1.5)
  // 10.0.2.2 là IP loopback của máy ảo Android Studio.
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
      cameras[0], 
      ResolutionPreset.low, 
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
    
    // Kết nối tới WebSocket của FastAPI
    _channel = WebSocketChannel.connect(Uri.parse(serverUrl));
    
    setState(() {
      _isStreaming = true;
      _latestWarning = "Đã kết nối Server. Đang quét...";
    });
    
    // Lắng nghe cảnh báo từ AI trả về
    _channel!.stream.listen((message) {
      final data = jsonDecode(message);
      if (data['success'] == true) {
        List objects = data['objects'] ?? [];
        if (objects.isNotEmpty) {
          List<String> warnings = objects.map((obj) => obj['class'].toString()).toList();
          String warningText = "Phía trước có ${warnings.join(' và ')}.";
          
          setState(() {
            _latestWarning = warningText;
          });
          
          flutterTts.speak(warningText);
        }
      }
    }, onDone: () {
      _stopStreaming();
    }, onError: (error) {
      setState(() {
        _latestWarning = "Lỗi kết nối: $error";
      });
      _stopStreaming();
    });

    // Cứ mỗi 1 giây sẽ chụp 1 bức ảnh (JPEG) gửi lên Server để phân tích
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) async {
      if (_isProcessingFrame || !_isStreaming) return;
      _isProcessingFrame = true;
      
      try {
        // Chụp ảnh định dạng JPEG
        XFile file = await _controller!.takePicture();
        
        // Chuyển file ảnh thành Base64 để gửi qua mạng
        List<int> imageBytes = await File(file.path).readAsBytes();
        String base64Image = base64Encode(imageBytes);
        
        // Gửi lên WebSocket
        _channel?.sink.add(base64Image); 
      } catch (e) {
        print("Lỗi chụp ảnh: $e");
      }
      
      _isProcessingFrame = false;
    });
  }

  void _stopStreaming() {
    setState(() {
      _isStreaming = false;
      if (_latestWarning.contains("Đang quét")) {
        _latestWarning = "Đã dừng quét.";
      }
    });
    _timer?.cancel();
    _channel?.sink.close();
  }

  @override
  void dispose() {
    _timer?.cancel();
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
