import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'dart:async';
import 'dart:typed_data';
import 'package:geolocator/geolocator.dart';
import '../services/api_service.dart';
import '../services/tts_service.dart';
import '../services/location_service.dart';

class CameraScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  
  const CameraScreen({super.key, required this.cameras});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  final ApiService _apiService = ApiService();
  final TtsService _ttsService = TtsService();
  final LocationService _locationService = LocationService();
  
  bool _isStreaming = false;
  bool _isProcessingFrame = false;
  String _latestWarning = "Bấm Bắt đầu để quét";
  String _currentLocation = "Đang tìm vệ tinh GPS...";
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _initCamera();
    _ttsService.init();
    _fetchLocation();
  }

  Future<void> _fetchLocation() async {
    Position? position = await _locationService.getCurrentLocation();
    if (position != null && mounted) {
      setState(() {
        _currentLocation = "GPS: ${position.latitude.toStringAsFixed(4)}, ${position.longitude.toStringAsFixed(4)}";
      });
    } else if (mounted) {
      setState(() {
        _currentLocation = "Không lấy được GPS";
      });
    }
  }

  Future<void> _initCamera() async {
    if (widget.cameras.isEmpty) return;
    _controller = CameraController(
      widget.cameras[0],
      ResolutionPreset.low,
      enableAudio: false,
    );
    await _controller!.initialize();
    if (mounted) setState(() {});
  }

  void _startStreaming() {
    if (_controller == null || !_controller!.value.isInitialized) return;
    
    // Bật mạng
    _apiService.connect(
      onGuidanceReceived: (text, priority) {
        setState(() {
          _latestWarning = text; // Hiển thị thẳng câu tiếng Việt hoàn chỉnh từ Backend
        });
        _ttsService.speak(text); // Đọc lên loa
      },
      onDone: _stopStreaming,
      onError: (error) {
        print("Lỗi WebSocket: $error");
        _stopStreaming();
      }
    );
    
    setState(() {
      _isStreaming = true;
      _latestWarning = "Đã kết nối Server. Đang quét...";
    });

    // Thay vì dùng startImageStream (khó convert YUV), ta dùng Timer để chụp ảnh liên tục
    // Mỗi 1.5 giây sẽ chộp 1 khung hình (JPEG) và gửi đi. Vừa nhẹ máy vừa dễ làm MVP.
    _timer = Timer.periodic(const Duration(milliseconds: 1500), (timer) async {
      if (_isProcessingFrame || !_isStreaming) return;
      _isProcessingFrame = true;
      
      try {
        // Chụp ảnh từ Camera
        XFile picture = await _controller!.takePicture();
        // Đọc thành mảng bytes
        Uint8List bytes = await picture.readAsBytes();
        
        // Gửi qua mạng
        _apiService.sendImage(bytes);
      } catch (e) {
        print("Lỗi chụp ảnh: $e");
      } finally {
        _isProcessingFrame = false;
      }
    });
  }

  void _stopStreaming() {
    _timer?.cancel();
    setState(() {
      _isStreaming = false;
      _latestWarning = "Đã dừng quét.";
    });
    // Không dùng stopImageStream nữa
    _apiService.disconnect();
  }

  @override
  void dispose() {
    _timer?.cancel();
    _controller?.dispose();
    _apiService.disconnect();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text("Kính Thông Minh AI", style: TextStyle(color: Colors.white)),
        backgroundColor: Colors.teal,
      ),
      body: Column(
        children: [
          Expanded(
            child: _controller == null || !_controller!.value.isInitialized
                ? const Center(child: CircularProgressIndicator(color: Colors.teal))
                : CameraPreview(_controller!),
          ),
          Container(
            padding: const EdgeInsets.all(20),
            decoration: const BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
            ),
            width: double.infinity,
            child: Column(
              children: [
                Text(
                  _currentLocation,
                  style: const TextStyle(fontSize: 14, color: Colors.blueGrey, fontStyle: FontStyle.italic),
                ),
                const SizedBox(height: 10),
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
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)),
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
