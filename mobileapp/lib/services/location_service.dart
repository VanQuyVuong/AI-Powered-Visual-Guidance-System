import 'package:geolocator/geolocator.dart';

/// Service lấy vị trí GPS
class LocationService {
  /// Kiểm tra quyền và lấy vị trí hiện tại
  Future<Position?> getCurrentLocation() async {
    bool serviceEnabled;
    LocationPermission permission;

    // Kiểm tra xem dịch vụ GPS trên máy có bật không
    try {
      serviceEnabled = await Geolocator.isLocationServiceEnabled();
      if (!serviceEnabled) {
        print('[GPS] Dịch vụ vị trí bị tắt.');
        return null;
      }
    } catch (e) {
      print('[GPS] Lỗi kiểm tra service: $e');
      return null;
    }

    permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) {
        print('[GPS] Người dùng từ chối cấp quyền vị trí.');
        return null;
      }
    }
    
    if (permission == LocationPermission.deniedForever) {
      print('[GPS] Quyền vị trí bị từ chối vĩnh viễn.');
      return null;
    } 

    // Khi đã có quyền, lấy vị trí hiện tại
    try {
      return await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      ).timeout(const Duration(seconds: 10));
    } catch (e) {
      print('[GPS] Lỗi lấy vị trí: $e');
      return null;
    }
  }
}
