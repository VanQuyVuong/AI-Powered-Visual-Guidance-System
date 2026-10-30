import 'package:flutter/material.dart';
import '../config/app_theme.dart';
import '../config/app_config.dart';
import '../services/api_service.dart';

/// Màn hình Cài đặt — cấu hình server, giọng nói, v.v.
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _ipController = TextEditingController();
  final _portController = TextEditingController();
  double _speechRate = AppConfig.defaultSpeechRate;
  bool _vibrationEnabled = AppConfig.defaultVibrationEnabled;
  int _scanInterval = AppConfig.defaultScanInterval;
  bool _isTesting = false;
  String? _testResult;

  @override
  void initState() {
    super.initState();
    _ipController.text = AppConfig.serverIp;
    _portController.text = AppConfig.serverPort;
    _speechRate = AppConfig.speechRate;
    _vibrationEnabled = AppConfig.vibrationEnabled;
    _scanInterval = AppConfig.scanInterval;
  }

  @override
  void dispose() {
    _ipController.dispose();
    _portController.dispose();
    super.dispose();
  }

  Future<void> _testConnection() async {
    setState(() {
      _isTesting = true;
      _testResult = null;
    });

    // Lưu tạm IP/Port để test
    AppConfig.serverIp = _ipController.text.trim();
    AppConfig.serverPort = _portController.text.trim();

    final api = ApiService();
    final success = await api.testConnection();

    if (mounted) {
      setState(() {
        _isTesting = false;
        _testResult = success ? 'Kết nối thành công!' : 'Không kết nối được. Kiểm tra IP và Backend.';
      });
    }
  }

  void _saveSettings() {
    AppConfig.serverIp = _ipController.text.trim();
    AppConfig.serverPort = _portController.text.trim();
    AppConfig.speechRate = _speechRate;
    AppConfig.vibrationEnabled = _vibrationEnabled;
    AppConfig.scanInterval = _scanInterval;

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: const Row(
          children: [
            Icon(Icons.check_circle, color: Colors.white),
            SizedBox(width: 12),
            Text('Đã lưu cài đặt!'),
          ],
        ),
        backgroundColor: AppTheme.success.withOpacity(0.8),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    );

    Navigator.of(context).pop(true); // true = đã thay đổi settings
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.bgDark,
      appBar: AppBar(
        backgroundColor: AppTheme.bgDark,
        title: const Text('Cài đặt', style: TextStyle(color: AppTheme.textPrimary)),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_rounded, color: AppTheme.textPrimary),
          onPressed: () => Navigator.of(context).pop(),
        ),
        actions: [
          TextButton(
            onPressed: _saveSettings,
            child: const Text('LƯU', style: TextStyle(color: AppTheme.primary, fontWeight: FontWeight.w700, fontSize: 16)),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(AppTheme.spacingMd),
        children: [
          // ═══════════════════════════════════════
          // SECTION: Kết nối Server
          // ═══════════════════════════════════════
          _buildSectionHeader('Kết nối Server', Icons.wifi_rounded),
          const SizedBox(height: 12),

          _buildCard(
            children: [
              _buildTextField(
                controller: _ipController,
                label: 'Địa chỉ IP máy tính',
                hint: '192.168.1.67',
                icon: Icons.computer_rounded,
                keyboardType: TextInputType.number,
              ),
              const SizedBox(height: 16),
              _buildTextField(
                controller: _portController,
                label: 'Cổng (Port)',
                hint: '8000',
                icon: Icons.numbers_rounded,
                keyboardType: TextInputType.number,
              ),
              const SizedBox(height: 16),

              // Nút test kết nối
              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: _isTesting ? null : _testConnection,
                  icon: _isTesting
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                        )
                      : const Icon(Icons.speed_rounded),
                  label: Text(_isTesting ? 'Đang kiểm tra...' : 'Kiểm tra kết nối'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.primary,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                ),
              ),

              if (_testResult != null) ...[
                const SizedBox(height: 12),
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: _testResult!.contains('thành công')
                        ? AppTheme.success.withOpacity(0.15)
                        : AppTheme.danger.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(
                      color: _testResult!.contains('thành công')
                          ? AppTheme.success.withOpacity(0.3)
                          : AppTheme.danger.withOpacity(0.3),
                    ),
                  ),
                  child: Row(
                    children: [
                      Icon(
                        _testResult!.contains('thành công')
                            ? Icons.check_circle_rounded
                            : Icons.error_rounded,
                        color: _testResult!.contains('thành công')
                            ? AppTheme.success
                            : AppTheme.danger,
                        size: 20,
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Text(
                          _testResult!,
                          style: TextStyle(
                            color: _testResult!.contains('thành công')
                                ? AppTheme.success
                                : AppTheme.danger,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ],
          ),

          const SizedBox(height: 24),

          // ═══════════════════════════════════════
          // SECTION: Giọng nói
          // ═══════════════════════════════════════
          _buildSectionHeader('Giọng nói', Icons.record_voice_over_rounded),
          const SizedBox(height: 12),

          _buildCard(
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text('Tốc độ đọc', style: TextStyle(color: AppTheme.textPrimary, fontSize: 16)),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                    decoration: BoxDecoration(
                      color: AppTheme.primary.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      '${(_speechRate * 100).round()}%',
                      style: const TextStyle(color: AppTheme.primary, fontWeight: FontWeight.w600),
                    ),
                  ),
                ],
              ),
              Slider(
                value: _speechRate,
                min: 0.1,
                max: 1.0,
                divisions: 9,
                activeColor: AppTheme.primary,
                inactiveColor: AppTheme.bgSurface,
                label: '${(_speechRate * 100).round()}%',
                onChanged: (value) => setState(() => _speechRate = value),
              ),
              const Text(
                'Chậm ←→ Nhanh',
                style: TextStyle(color: AppTheme.textMuted, fontSize: 12),
                textAlign: TextAlign.center,
              ),
            ],
          ),

          const SizedBox(height: 24),

          // ═══════════════════════════════════════
          // SECTION: Cảnh báo
          // ═══════════════════════════════════════
          _buildSectionHeader('Cảnh báo', Icons.notifications_active_rounded),
          const SizedBox(height: 12),

          _buildCard(
            children: [
              SwitchListTile(
                title: const Text('Rung cảnh báo', style: TextStyle(color: AppTheme.textPrimary)),
                subtitle: const Text('Rung khi phát hiện vật cản nguy hiểm', style: TextStyle(color: AppTheme.textSecondary, fontSize: 13)),
                value: _vibrationEnabled,
                activeColor: AppTheme.primary,
                contentPadding: EdgeInsets.zero,
                onChanged: (value) => setState(() => _vibrationEnabled = value),
              ),
              const Divider(color: AppTheme.bgSurface, height: 24),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Tần suất quét', style: TextStyle(color: AppTheme.textPrimary, fontSize: 16)),
                        SizedBox(height: 4),
                        Text('Thời gian giữa mỗi lần chụp ảnh', style: TextStyle(color: AppTheme.textSecondary, fontSize: 13)),
                      ],
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                    decoration: BoxDecoration(
                      color: AppTheme.accent.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      '${(_scanInterval / 1000).toStringAsFixed(1)}s',
                      style: const TextStyle(color: AppTheme.accent, fontWeight: FontWeight.w600),
                    ),
                  ),
                ],
              ),
              Slider(
                value: _scanInterval.toDouble(),
                min: 500,
                max: 3000,
                divisions: 5,
                activeColor: AppTheme.accent,
                inactiveColor: AppTheme.bgSurface,
                label: '${(_scanInterval / 1000).toStringAsFixed(1)}s',
                onChanged: (value) => setState(() => _scanInterval = value.round()),
              ),
              const Text(
                'Nhanh (0.5s) ←→ Tiết kiệm pin (3s)',
                style: TextStyle(color: AppTheme.textMuted, fontSize: 12),
                textAlign: TextAlign.center,
              ),
            ],
          ),

          const SizedBox(height: 24),

          // ═══════════════════════════════════════
          // SECTION: Thông tin
          // ═══════════════════════════════════════
          _buildSectionHeader('Thông tin', Icons.info_outline_rounded),
          const SizedBox(height: 12),

          _buildCard(
            children: [
              _buildInfoRow('Phiên bản', 'v1.0.0'),
              const Divider(color: AppTheme.bgSurface, height: 20),
              _buildInfoRow('Server URL', AppConfig.wsUrl),
              const Divider(color: AppTheme.bgSurface, height: 20),
              _buildInfoRow('Dự án', 'AI Visual Guidance System'),
            ],
          ),

          const SizedBox(height: 40),
        ],
      ),
    );
  }

  // ── Helper Widgets ──

  Widget _buildSectionHeader(String title, IconData icon) {
    return Row(
      children: [
        Icon(icon, color: AppTheme.primary, size: 22),
        const SizedBox(width: 10),
        Text(
          title,
          style: const TextStyle(
            color: AppTheme.textPrimary,
            fontSize: 18,
            fontWeight: FontWeight.w600,
          ),
        ),
      ],
    );
  }

  Widget _buildCard({required List<Widget> children}) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppTheme.bgCard,
        borderRadius: BorderRadius.circular(AppTheme.radiusMd),
        border: Border.all(color: AppTheme.bgSurface.withOpacity(0.5)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: children,
      ),
    );
  }

  Widget _buildTextField({
    required TextEditingController controller,
    required String label,
    required String hint,
    required IconData icon,
    TextInputType? keyboardType,
  }) {
    return TextField(
      controller: controller,
      keyboardType: keyboardType,
      style: const TextStyle(color: AppTheme.textPrimary, fontSize: 16),
      decoration: InputDecoration(
        labelText: label,
        hintText: hint,
        labelStyle: const TextStyle(color: AppTheme.textSecondary),
        hintStyle: const TextStyle(color: AppTheme.textMuted),
        prefixIcon: Icon(icon, color: AppTheme.primary),
        filled: true,
        fillColor: AppTheme.bgSurface,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide.none,
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: AppTheme.primary, width: 2),
        ),
      ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(label, style: const TextStyle(color: AppTheme.textSecondary)),
        Flexible(
          child: Text(
            value,
            style: const TextStyle(color: AppTheme.textPrimary, fontWeight: FontWeight.w500),
            overflow: TextOverflow.ellipsis,
          ),
        ),
      ],
    );
  }
}
