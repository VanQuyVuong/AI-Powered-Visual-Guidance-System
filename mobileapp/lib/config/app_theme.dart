import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

/// Design system — màu sắc, typography, và constants cho toàn bộ app.
/// Thiết kế tối ưu cho người khiếm thị: tương phản cao, font lớn.
class AppTheme {
  // ── Bảng màu chính ──
  static const Color primary = Color(0xFF00897B);        // Teal đậm
  static const Color primaryLight = Color(0xFF4DB6AC);    // Teal nhạt
  static const Color primaryDark = Color(0xFF00695C);     // Teal rất đậm

  static const Color accent = Color(0xFFFFB300);          // Amber cho cảnh báo
  static const Color accentLight = Color(0xFFFFD54F);

  static const Color danger = Color(0xFFFF5252);          // Đỏ nguy hiểm
  static const Color dangerDark = Color(0xFFD32F2F);
  static const Color success = Color(0xFF69F0AE);         // Xanh lá an toàn
  static const Color info = Color(0xFF42A5F5);            // Xanh dương thông tin

  // ── Nền tối ──
  static const Color bgDark = Color(0xFF0D1117);          // Nền chính
  static const Color bgCard = Color(0xFF161B22);          // Nền card
  static const Color bgSurface = Color(0xFF21262D);       // Nền surface
  static const Color bgOverlay = Color(0x99000000);       // Overlay bán trong suốt

  // ── Text ──
  static const Color textPrimary = Color(0xFFF0F6FC);     // Trắng chính
  static const Color textSecondary = Color(0xFF8B949E);   // Xám phụ
  static const Color textMuted = Color(0xFF484F58);       // Xám nhạt

  // ── Gradient ──
  static const LinearGradient primaryGradient = LinearGradient(
    colors: [primary, Color(0xFF26A69A)],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );

  static const LinearGradient dangerGradient = LinearGradient(
    colors: [danger, Color(0xFFFF8A80)],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );

  static const LinearGradient cardGradient = LinearGradient(
    colors: [Color(0xFF1A2332), Color(0xFF0D1117)],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );

  // ── Border Radius ──
  static const double radiusSm = 8;
  static const double radiusMd = 16;
  static const double radiusLg = 24;
  static const double radiusXl = 32;

  // ── Spacing ──
  static const double spacingSm = 8;
  static const double spacingMd = 16;
  static const double spacingLg = 24;
  static const double spacingXl = 32;

  /// ThemeData cho MaterialApp
  static ThemeData get darkTheme {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: bgDark,
      colorScheme: const ColorScheme.dark(
        primary: primary,
        secondary: accent,
        surface: bgSurface,
        error: danger,
        onPrimary: Colors.white,
        onSecondary: Colors.black,
        onSurface: textPrimary,
      ),
      textTheme: GoogleFonts.interTextTheme(
        const TextTheme(
          headlineLarge: TextStyle(
            fontSize: 32,
            fontWeight: FontWeight.w700,
            color: textPrimary,
            letterSpacing: -0.5,
          ),
          headlineMedium: TextStyle(
            fontSize: 24,
            fontWeight: FontWeight.w600,
            color: textPrimary,
          ),
          titleLarge: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.w600,
            color: textPrimary,
          ),
          titleMedium: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w500,
            color: textPrimary,
          ),
          bodyLarge: TextStyle(
            fontSize: 16,
            color: textPrimary,
          ),
          bodyMedium: TextStyle(
            fontSize: 14,
            color: textSecondary,
          ),
          labelLarge: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: textPrimary,
          ),
        ),
      ),
      appBarTheme: AppBarTheme(
        backgroundColor: bgDark,
        elevation: 0,
        centerTitle: true,
        titleTextStyle: GoogleFonts.inter(
          fontSize: 20,
          fontWeight: FontWeight.w600,
          color: textPrimary,
        ),
        iconTheme: const IconThemeData(color: textPrimary),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: primary,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(radiusLg),
          ),
          textStyle: GoogleFonts.inter(
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
    );
  }

  // ── Helpers ──

  /// Lấy màu theo mức độ ưu tiên cảnh báo
  static Color priorityColor(String priority) {
    switch (priority.toUpperCase()) {
      case 'HIGH':
        return danger;
      case 'MEDIUM':
        return accent;
      case 'LOW':
        return info;
      default:
        return textSecondary;
    }
  }

  /// Lấy icon theo mức độ ưu tiên
  static IconData priorityIcon(String priority) {
    switch (priority.toUpperCase()) {
      case 'HIGH':
        return Icons.warning_rounded;
      case 'MEDIUM':
        return Icons.info_rounded;
      case 'LOW':
        return Icons.check_circle_rounded;
      default:
        return Icons.remove_circle_outline;
    }
  }
}
