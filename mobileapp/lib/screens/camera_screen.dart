import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'dart:async';
import 'dart:typed_data';
import '../config/app_theme.dart';
import '../config/app_config.dart';
import '../services/api_service.dart';
import '../services/tts_service.dart';
import '../services/audio_service.dart';
import '../services/location_service.dart';
import '../services/voice_service.dart';
import 'settings_screen.dart';

class CameraScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  const CameraScreen({super.key, required this.cameras});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> with TickerProviderStateMixin {
  // ── Controllers & Services ──
  CameraController? _cameraController;
  final ApiService _apiService = ApiService();
  final TtsService _ttsService = TtsService();
  final AudioService _audioService = AudioService();
  final LocationService _locationService = LocationService();
  final VoiceService _voiceService = VoiceService();

  // ── State ──
  bool _isStreaming = false;
  bool _isProcessingFrame = false;
  bool _isListening = false;
  bool _isReadingText = false;
  String _currentMode = 'NAVIGATION'; // NAVIGATION hoặc CONVERSATION

  // ── Data từ Backend ──
  String _warningText = '';
  String _warningPriority = 'NONE';
  String _navigationDirection = '';
  String _navigationText = '';
  int _objectCount = 0;
  ConnectionStatus _connectionStatus = ConnectionStatus.disconnected;

  // ── Location ──
  String _locationText = '';

  // ── Timer ──
  Timer? _scanTimer;

  // ── Animations ──
  late AnimationController _warningAnimController;
  late Animation<double> _warningSlideAnimation;
  late AnimationController _pulseAnimController;
  late Animation<double> _pulseAnimation;
  late AnimationController _navArrowAnimController;
  late Animation<double> _navArrowAnimation;

  @override
  void initState() {
    super.initState();

    // Warning slide animation
    _warningAnimController = AnimationController(
      duration: const Duration(milliseconds: 400),
      vsync: this,
    );
    _warningSlideAnimation = Tween<double>(begin: -100, end: 0).animate(
      CurvedAnimation(parent: _warningAnimController, curve: Curves.easeOutCubic),
    );

    // Pulse animation cho nút scan
    _pulseAnimController = AnimationController(
      duration: const Duration(milliseconds: 1200),
      vsync: this,
    );
    _pulseAnimation = Tween<double>(begin: 1.0, end: 1.15).animate(
      CurvedAnimation(parent: _pulseAnimController, curve: Curves.easeInOut),
    );

    // Navigation arrow bob animation
    _navArrowAnimController = AnimationController(
      duration: const Duration(milliseconds: 800),
      vsync: this,
    )..repeat(reverse: true);
    _navArrowAnimation = Tween<double>(begin: -8, end: 8).animate(
      CurvedAnimation(parent: _navArrowAnimController, curve: Curves.easeInOut),
    );

    _initAll();
  }

  Future<void> _initAll() async {
    await _initCamera();
    await _ttsService.init();
    await _audioService.init();
    await _voiceService.init();
    _fetchLocation();
  }

  Future<void> _initCamera() async {
    if (widget.cameras.isEmpty) return;
    _cameraController = CameraController(
      widget.cameras[0],
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );
    try {
      await _cameraController!.initialize();
      if (mounted) setState(() {});
    } catch (e) {
      print('[Camera] Lỗi khởi tạo: $e');
    }
  }

  Future<void> _fetchLocation() async {
    try {
      final position = await _locationService.getCurrentLocation();
      if (position != null && mounted) {
        setState(() {
          _locationText = '${position.latitude.toStringAsFixed(4)}, ${position.longitude.toStringAsFixed(4)}';
        });
      }
    } catch (_) {
      if (mounted) setState(() => _locationText = 'Không có GPS');
    }
  }

  // ════════════════════════════════════════════
  // STREAMING
  // ════════════════════════════════════════════

  void _startStreaming() {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      _ttsService.speak('Không tìm thấy camera');
      return;
    }

    _apiService.connect(
      onGuidanceReceived: (text, priority, fullData) {
        if (!mounted) return;
        setState(() {
          _warningText = text;
          _warningPriority = priority;
          _objectCount = (fullData['objects'] as List?)?.length ?? 0;
        });

        // Hiện warning overlay
        if (text.isNotEmpty) {
          _warningAnimController.forward();
        }

        // Rung nếu nguy hiểm
        if (priority == 'HIGH' && AppConfig.vibrationEnabled) {
          HapticFeedback.heavyImpact();
        } else if (priority == 'MEDIUM' && AppConfig.vibrationEnabled) {
          HapticFeedback.mediumImpact();
        }

        // Đọc cảnh báo bằng giọng nói
        _ttsService.speak(text);
      },
      onAudioReceived: (audioBase64) {
        if (audioBase64 != null && audioBase64.isNotEmpty) {
          _audioService.playBase64Audio(audioBase64);
        }
      },
      onNavigationReceived: (nav) {
        if (!mounted) return;
        setState(() {
          _navigationDirection = (nav['direction'] ?? '').toString();
          _navigationText = (nav['text'] ?? '').toString();
        });
      },
      onStatusChanged: (status) {
        if (!mounted) return;
        setState(() => _connectionStatus = status);
      },
      onDone: () {
        if (mounted) _stopStreaming();
      },
      onError: (error) {
        print('[Stream] Lỗi: $error');
      },
    );

    setState(() {
      _isStreaming = true;
      _warningText = '';
      _warningPriority = 'NONE';
    });

    _pulseAnimController.repeat(reverse: true);
    _ttsService.speak('Bắt đầu quét môi trường');

    // Chụp ảnh định kỳ và gửi lên backend
    _scanTimer = Timer.periodic(
      Duration(milliseconds: AppConfig.scanInterval),
      (timer) async {
        if (_isProcessingFrame || !_isStreaming) return;
        _isProcessingFrame = true;

        try {
          XFile picture = await _cameraController!.takePicture();
          Uint8List bytes = await picture.readAsBytes();
          _apiService.sendImage(bytes);
        } catch (e) {
          print('[Camera] Lỗi chụp: $e');
        } finally {
          _isProcessingFrame = false;
        }
      },
    );
  }

  void _stopStreaming() {
    _scanTimer?.cancel();
    _apiService.disconnect();
    _pulseAnimController.stop();
    _pulseAnimController.reset();

    if (mounted) {
      setState(() {
        _isStreaming = false;
        _connectionStatus = ConnectionStatus.disconnected;
      });
      _warningAnimController.reverse();
      _ttsService.speak('Đã dừng quét');
    }
  }

  // ════════════════════════════════════════════
  // OCR
  // ════════════════════════════════════════════

  Future<void> _readText() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) return;
    if (_isProcessingFrame || _isReadingText) return;

    setState(() => _isReadingText = true);
    _ttsService.speak('Đang phân tích chữ viết');

    try {
      _isProcessingFrame = true;
      XFile picture = await _cameraController!.takePicture();
      Uint8List bytes = await picture.readAsBytes();

      String? text = await _apiService.readTextFromImage(bytes);

      if (text != null && text.isNotEmpty && mounted) {
        setState(() {
          _warningText = '📖 $text';
          _warningPriority = 'LOW';
        });
        _warningAnimController.forward();
        _ttsService.speak(text);
      } else {
        _ttsService.speak('Không tìm thấy chữ nào');
      }
    } catch (e) {
      print('[OCR] Lỗi: $e');
      _ttsService.speak('Lỗi khi đọc chữ');
    } finally {
      _isProcessingFrame = false;
      if (mounted) setState(() => _isReadingText = false);
    }
  }

  // ════════════════════════════════════════════
  // VOICE COMMAND
  // ════════════════════════════════════════════

  void _toggleListening() {
    if (_isListening) {
      _voiceService.stopListening();
      setState(() => _isListening = false);
    } else {
      setState(() => _isListening = true);
      _ttsService.speak('Đang nghe lệnh');

      _voiceService.startListening((text) {
        final lower = text.toLowerCase();
        if (lower.contains('bắt đầu') || lower.contains('quét')) {
          if (!_isStreaming) _startStreaming();
          _toggleListening();
        } else if (lower.contains('dừng') || lower.contains('tắt')) {
          if (_isStreaming) _stopStreaming();
          _toggleListening();
        } else if (lower.contains('đọc') || lower.contains('chữ')) {
          _readText();
          _toggleListening();
        } else if (lower.contains('hỏi') || lower.contains('là gì')) {
          _handleQuestion(text);
          _toggleListening();
        }
      });
    }
  }

  Future<void> _handleQuestion(String question) async {
    _ttsService.speak('Đang xử lý câu hỏi');
    String? answer = await _apiService.askQuestion(question, []);
    if (answer != null && answer.isNotEmpty) {
      _ttsService.speak(answer);
      if (mounted) {
        setState(() {
          _warningText = answer;
          _warningPriority = 'LOW';
        });
        _warningAnimController.forward();
      }
    }
  }

  // ════════════════════════════════════════════
  // MODE SWITCH
  // ════════════════════════════════════════════

  Future<void> _toggleMode() async {
    final newMode = _currentMode == 'NAVIGATION' ? 'CONVERSATION' : 'NAVIGATION';
    String? message = await _apiService.setMode(newMode);
    if (mounted) {
      setState(() => _currentMode = newMode);
      if (message != null) {
        _ttsService.speak(message);
      }
    }
  }

  @override
  void dispose() {
    _scanTimer?.cancel();
    _cameraController?.dispose();
    _apiService.disconnect();
    _audioService.dispose();
    _warningAnimController.dispose();
    _pulseAnimController.dispose();
    _navArrowAnimController.dispose();
    super.dispose();
  }

  // ════════════════════════════════════════════
  // BUILD UI
  // ════════════════════════════════════════════

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.bgDark,
      body: Stack(
        children: [
          // ── Layer 1: Camera Preview (full screen) ──
          _buildCameraPreview(),

          // ── Layer 2: Top Status Bar ──
          _buildTopBar(),

          // ── Layer 3: Navigation Direction Indicator ──
          if (_isStreaming && _navigationDirection.isNotEmpty)
            _buildNavigationIndicator(),

          // ── Layer 4: Warning Overlay ──
          if (_warningText.isNotEmpty)
            _buildWarningOverlay(),

          // ── Layer 5: Bottom Control Panel ──
          _buildBottomPanel(),

          // ── Layer 6: Voice Listening Overlay ──
          if (_isListening)
            _buildListeningOverlay(),
        ],
      ),
    );
  }

  // ── Camera Preview ──
  Widget _buildCameraPreview() {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return Container(
        color: AppTheme.bgDark,
        child: const Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              CircularProgressIndicator(color: AppTheme.primary),
              SizedBox(height: 16),
              Text('Đang khởi tạo camera...',
                  style: TextStyle(color: AppTheme.textSecondary)),
            ],
          ),
        ),
      );
    }

    return SizedBox.expand(
      child: FittedBox(
        fit: BoxFit.cover,
        child: SizedBox(
          width: _cameraController!.value.previewSize!.height,
          height: _cameraController!.value.previewSize!.width,
          child: CameraPreview(_cameraController!),
        ),
      ),
    );
  }

  // ── Top Status Bar ──
  Widget _buildTopBar() {
    return Positioned(
      top: 0,
      left: 0,
      right: 0,
      child: Container(
        padding: EdgeInsets.only(
          top: MediaQuery.of(context).padding.top + 8,
          left: 16,
          right: 16,
          bottom: 12,
        ),
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xCC000000), Color(0x00000000)],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: Row(
          children: [
            // Logo + Tên app
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: AppTheme.bgCard.withOpacity(0.8),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: AppTheme.bgSurface),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.visibility_rounded, color: AppTheme.primary, size: 18),
                  const SizedBox(width: 6),
                  Text(
                    'AI Guide',
                    style: TextStyle(
                      color: AppTheme.textPrimary,
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ),

            const Spacer(),

            // Connection Status
            _buildStatusChip(
              icon: _connectionStatusIcon(),
              label: _connectionStatusText(),
              color: _connectionStatusColor(),
            ),

            const SizedBox(width: 8),

            // Mode indicator
            GestureDetector(
              onTap: _toggleMode,
              child: _buildStatusChip(
                icon: _currentMode == 'NAVIGATION'
                    ? Icons.navigation_rounded
                    : Icons.chat_rounded,
                label: _currentMode == 'NAVIGATION' ? 'Dẫn đường' : 'Hỏi đáp',
                color: _currentMode == 'NAVIGATION'
                    ? AppTheme.primary
                    : AppTheme.accent,
              ),
            ),

            const SizedBox(width: 8),

            // Settings button
            GestureDetector(
              onTap: () async {
                final changed = await Navigator.of(context).push<bool>(
                  MaterialPageRoute(builder: (_) => const SettingsScreen()),
                );
                if (changed == true && mounted) {
                  _ttsService.updateSpeechRate(AppConfig.speechRate);
                }
              },
              child: Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: AppTheme.bgCard.withOpacity(0.8),
                  shape: BoxShape.circle,
                  border: Border.all(color: AppTheme.bgSurface),
                ),
                child: const Icon(Icons.settings_rounded, color: AppTheme.textSecondary, size: 20),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatusChip({required IconData icon, required String label, required Color color}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: color.withOpacity(0.15),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: color, size: 14),
          const SizedBox(width: 4),
          Text(label, style: TextStyle(color: color, fontSize: 11, fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }

  // ── Navigation Direction Indicator ──
  Widget _buildNavigationIndicator() {
    IconData dirIcon;
    Alignment alignment;
    double rotation = 0;

    switch (_navigationDirection.toUpperCase()) {
      case 'LEFT':
        dirIcon = Icons.arrow_back_rounded;
        alignment = Alignment.centerLeft;
        break;
      case 'RIGHT':
        dirIcon = Icons.arrow_forward_rounded;
        alignment = Alignment.centerRight;
        break;
      case 'STRAIGHT':
        dirIcon = Icons.arrow_upward_rounded;
        alignment = Alignment.center;
        break;
      case 'STOP':
        dirIcon = Icons.front_hand_rounded;
        alignment = Alignment.center;
        break;
      default:
        return const SizedBox.shrink();
    }

    return Positioned.fill(
      child: AnimatedBuilder(
        animation: _navArrowAnimation,
        builder: (context, child) {
          double dx = 0, dy = 0;
          if (_navigationDirection.toUpperCase() == 'LEFT') {
            dx = _navArrowAnimation.value;
          } else if (_navigationDirection.toUpperCase() == 'RIGHT') {
            dx = -_navArrowAnimation.value;
          } else {
            dy = _navArrowAnimation.value;
          }

          return Align(
            alignment: alignment,
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Transform.translate(
                offset: Offset(dx, dy),
                child: Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: _navigationDirection.toUpperCase() == 'STOP'
                        ? AppTheme.danger.withOpacity(0.3)
                        : AppTheme.primary.withOpacity(0.3),
                    shape: BoxShape.circle,
                    border: Border.all(
                      color: _navigationDirection.toUpperCase() == 'STOP'
                          ? AppTheme.danger.withOpacity(0.6)
                          : AppTheme.primary.withOpacity(0.6),
                      width: 2,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: (_navigationDirection.toUpperCase() == 'STOP'
                                ? AppTheme.danger
                                : AppTheme.primary)
                            .withOpacity(0.3),
                        blurRadius: 20,
                        spreadRadius: 4,
                      ),
                    ],
                  ),
                  child: Icon(
                    dirIcon,
                    color: Colors.white,
                    size: 40,
                  ),
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  // ── Warning Overlay ──
  Widget _buildWarningOverlay() {
    final color = AppTheme.priorityColor(_warningPriority);
    final icon = AppTheme.priorityIcon(_warningPriority);

    return AnimatedBuilder(
      animation: _warningSlideAnimation,
      builder: (context, child) {
        return Positioned(
          top: MediaQuery.of(context).padding.top + 70,
          left: 16,
          right: 16,
          child: Transform.translate(
            offset: Offset(0, _warningSlideAnimation.value),
            child: GestureDetector(
              onTap: () {
                _warningAnimController.reverse();
                Future.delayed(const Duration(milliseconds: 400), () {
                  if (mounted) setState(() => _warningText = '');
                });
              },
              child: Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(AppTheme.radiusMd),
                  border: Border.all(color: color.withOpacity(0.4)),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.3),
                      blurRadius: 20,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(AppTheme.radiusMd),
                  child: Container(
                    color: Colors.transparent,
                    child: Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            color: color.withOpacity(0.2),
                            shape: BoxShape.circle,
                          ),
                          child: Icon(icon, color: color, size: 24),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                _priorityLabel(_warningPriority),
                                style: TextStyle(
                                  color: color,
                                  fontSize: 12,
                                  fontWeight: FontWeight.w700,
                                  letterSpacing: 0.5,
                                ),
                              ),
                              const SizedBox(height: 4),
                              Text(
                                _warningText,
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 16,
                                  fontWeight: FontWeight.w500,
                                ),
                                maxLines: 3,
                                overflow: TextOverflow.ellipsis,
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  // ── Bottom Control Panel ──
  Widget _buildBottomPanel() {
    return Positioned(
      bottom: 0,
      left: 0,
      right: 0,
      child: Container(
        padding: EdgeInsets.only(
          left: 20,
          right: 20,
          top: 20,
          bottom: MediaQuery.of(context).padding.bottom + 20,
        ),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [
              Colors.transparent,
              AppTheme.bgDark.withOpacity(0.7),
              AppTheme.bgDark.withOpacity(0.95),
            ],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // ── Info Row ──
            if (_isStreaming)
              Padding(
                padding: const EdgeInsets.only(bottom: 16),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    _buildInfoChip(Icons.radar_rounded, '$_objectCount vật thể'),
                    const SizedBox(width: 12),
                    if (_locationText.isNotEmpty)
                      _buildInfoChip(Icons.location_on_rounded, _locationText),
                  ],
                ),
              ),

            // ── Main Buttons Row ──
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                // OCR Button
                _buildCircleButton(
                  icon: Icons.menu_book_rounded,
                  label: 'Đọc chữ',
                  color: AppTheme.info,
                  isLoading: _isReadingText,
                  onTap: _readText,
                  size: 56,
                ),

                // Main Scan Button
                _buildMainScanButton(),

                // Voice Button
                _buildCircleButton(
                  icon: _isListening ? Icons.mic_rounded : Icons.mic_none_rounded,
                  label: _isListening ? 'Đang nghe' : 'Giọng nói',
                  color: _isListening ? AppTheme.danger : AppTheme.accent,
                  onTap: _toggleListening,
                  size: 56,
                  isActive: _isListening,
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMainScanButton() {
    return GestureDetector(
      onTap: _isStreaming ? _stopStreaming : _startStreaming,
      child: AnimatedBuilder(
        animation: _pulseAnimation,
        builder: (context, child) {
          final scale = _isStreaming ? _pulseAnimation.value : 1.0;
          return Transform.scale(
            scale: scale,
            child: Container(
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                gradient: _isStreaming
                    ? AppTheme.dangerGradient
                    : AppTheme.primaryGradient,
                shape: BoxShape.circle,
                boxShadow: [
                  BoxShadow(
                    color: (_isStreaming ? AppTheme.danger : AppTheme.primary)
                        .withOpacity(0.4),
                    blurRadius: 20,
                    spreadRadius: 4,
                  ),
                ],
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(
                    _isStreaming ? Icons.stop_rounded : Icons.play_arrow_rounded,
                    color: Colors.white,
                    size: 36,
                  ),
                  Text(
                    _isStreaming ? 'DỪNG' : 'QUÉT',
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 10,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 1,
                    ),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildCircleButton({
    required IconData icon,
    required String label,
    required Color color,
    required VoidCallback onTap,
    double size = 52,
    bool isLoading = false,
    bool isActive = false,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            width: size,
            height: size,
            decoration: BoxDecoration(
              color: isActive
                  ? color.withOpacity(0.3)
                  : AppTheme.bgCard.withOpacity(0.8),
              shape: BoxShape.circle,
              border: Border.all(
                color: isActive ? color : color.withOpacity(0.4),
                width: isActive ? 2 : 1,
              ),
              boxShadow: isActive
                  ? [BoxShadow(color: color.withOpacity(0.3), blurRadius: 12)]
                  : null,
            ),
            child: isLoading
                ? Padding(
                    padding: const EdgeInsets.all(16),
                    child: CircularProgressIndicator(strokeWidth: 2, color: color),
                  )
                : Icon(icon, color: color, size: size * 0.45),
          ),
          const SizedBox(height: 6),
          Text(
            label,
            style: TextStyle(color: AppTheme.textSecondary, fontSize: 11),
          ),
        ],
      ),
    );
  }

  Widget _buildInfoChip(IconData icon, String text) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: AppTheme.bgCard.withOpacity(0.8),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.bgSurface.withOpacity(0.5)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: AppTheme.textSecondary, size: 14),
          const SizedBox(width: 4),
          Text(text, style: const TextStyle(color: AppTheme.textSecondary, fontSize: 11)),
        ],
      ),
    );
  }

  // ── Listening Overlay ──
  Widget _buildListeningOverlay() {
    return Positioned.fill(
      child: Container(
        color: Colors.black.withOpacity(0.6),
        child: Center(
          child: Container(
            padding: const EdgeInsets.all(40),
            decoration: BoxDecoration(
              color: AppTheme.bgCard,
              borderRadius: BorderRadius.circular(AppTheme.radiusLg),
              border: Border.all(color: AppTheme.danger.withOpacity(0.3)),
              boxShadow: [
                BoxShadow(
                  color: AppTheme.danger.withOpacity(0.2),
                  blurRadius: 30,
                  spreadRadius: 5,
                ),
              ],
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: AppTheme.danger.withOpacity(0.15),
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(Icons.mic_rounded, color: AppTheme.danger, size: 48),
                ),
                const SizedBox(height: 20),
                const Text(
                  'Đang nghe...',
                  style: TextStyle(
                    color: AppTheme.textPrimary,
                    fontSize: 22,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 8),
                const Text(
                  'Nói "bắt đầu", "dừng", "đọc chữ"',
                  style: TextStyle(color: AppTheme.textSecondary, fontSize: 14),
                ),
                const SizedBox(height: 20),
                TextButton(
                  onPressed: _toggleListening,
                  child: const Text('Huỷ', style: TextStyle(color: AppTheme.danger, fontSize: 16)),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  // ── Helpers ──

  String _priorityLabel(String priority) {
    switch (priority.toUpperCase()) {
      case 'HIGH':
        return '⚠️ NGUY HIỂM';
      case 'MEDIUM':
        return '⚡ CHÚ Ý';
      case 'LOW':
        return 'ℹ️ THÔNG TIN';
      default:
        return '';
    }
  }

  IconData _connectionStatusIcon() {
    switch (_connectionStatus) {
      case ConnectionStatus.connected:
        return Icons.wifi_rounded;
      case ConnectionStatus.connecting:
        return Icons.wifi_find_rounded;
      case ConnectionStatus.error:
        return Icons.wifi_off_rounded;
      case ConnectionStatus.disconnected:
        return Icons.wifi_off_rounded;
    }
  }

  String _connectionStatusText() {
    switch (_connectionStatus) {
      case ConnectionStatus.connected:
        return 'Đã nối';
      case ConnectionStatus.connecting:
        return 'Đang nối...';
      case ConnectionStatus.error:
        return 'Lỗi';
      case ConnectionStatus.disconnected:
        return 'Ngắt';
    }
  }

  Color _connectionStatusColor() {
    switch (_connectionStatus) {
      case ConnectionStatus.connected:
        return AppTheme.success;
      case ConnectionStatus.connecting:
        return AppTheme.accent;
      case ConnectionStatus.error:
        return AppTheme.danger;
      case ConnectionStatus.disconnected:
        return AppTheme.textMuted;
    }
  }
}
