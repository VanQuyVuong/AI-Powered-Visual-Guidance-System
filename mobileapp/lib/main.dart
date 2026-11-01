import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'config/app_theme.dart';
import 'config/app_config.dart';
import 'screens/splash_screen.dart';

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Cấu hình thanh status bar trong suốt
  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.light,
    systemNavigationBarColor: AppTheme.bgDark,
    systemNavigationBarIconBrightness: Brightness.light,
  ));

  // Cho phép xoay dọc
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
  ]);

  // Khởi tạo config (SharedPreferences)
  await AppConfig.init();

  // Lấy danh sách camera
  try {
    cameras = await availableCameras();
  } on CameraException catch (e) {
    print('Lỗi khởi tạo camera: $e');
  }

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Kính Thông Minh AI',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.darkTheme,
      home: SplashScreen(cameras: cameras),
    );
  }
}
