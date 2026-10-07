import 'package:geolocator/geolocator.dart';

class LocationService {
  /// Kiểm tra quyền và lấy vị trí hiện tại
  Future<Position?> getCurrentLocation() async {
    bool serviceEnabled;
    LocationPermission permission;

    // Kiểm tra xem dịch vụ GPS trên máy có bật không
    serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      print('Dịch vụ vị trí bị tắt.');
      return null;
    }

    permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) {
        print('Người dùng từ chối cấp quyền vị trí.');
        return null;
      }
    }
    
    if (permission == LocationPermission.deniedForever) {
      print('Quyền vị trí bị từ chối vĩnh viễn.');
      return null;
    } 

    // Khi đã có quyền, lấy vị trí hiện tại (Độ chính xác cao)
    return await Geolocator.getCurrentPosition(desiredAccuracy: LocationAccuracy.high);
  }
}
