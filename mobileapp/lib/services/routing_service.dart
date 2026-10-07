import 'dart:convert';
import 'package:http/http.dart' as http;

class RoutingService {
  // Thay thế bằng API Key của OpenRouteService
  final String _apiKey = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjQ1NWU2ZTdjMzAwZDQ5MDRhMjYyZDEzZjJmNzUyNDQ5IiwiaCI6Im11cm11cjY0In0=";
  
  /// Gọi API tìm đường đi bộ từ điểm A đến điểm B
  /// Trả về danh sách các bước chỉ đường (dạng Text tiếng Việt)
  Future<List<String>> getWalkingRoute(double startLat, double startLng, double endLat, double endLng) async {
    // OpenRouteService nhận tọa độ theo chuẩn [Kinh độ (Longitude), Vĩ độ (Latitude)]
    final String url = "https://api.openrouteservice.org/v2/directions/foot-walking"
        "?api_key=$_apiKey"
        "&start=$startLng,$startLat"
        "&end=$endLng,$endLat";

    try {
      final response = await http.get(Uri.parse(url));

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        
        // Trích xuất mảng hướng dẫn đi đường (steps)
        final features = data['features'];
        if (features != null && features.isNotEmpty) {
          final segments = features[0]['properties']['segments'];
          if (segments != null && segments.isNotEmpty) {
            final steps = segments[0]['steps'];
            List<String> instructions = [];
            
            for (var step in steps) {
              String instruction = step['instruction'];
              // ORS mặc định trả về tiếng Anh, ta có thể dùng AI để dịch hoặc 
              // tạm thời giữ nguyên để test logic. (Thường sẽ trả về "Turn left onto...", "Head straight...")
              instructions.add(instruction);
            }
            return instructions;
          }
        }
      } else {
        print("Lỗi API tìm đường: ${response.statusCode} - ${response.body}");
      }
    } catch (e) {
      print("Lỗi gọi API OpenRouteService: $e");
    }
    
    return [];
  }
}
