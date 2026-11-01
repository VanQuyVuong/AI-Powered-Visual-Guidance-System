"""
Dịch vụ trợ lý hội thoại AI cho người khiếm thị (Voice Assistant Service)
Trả lời thắc mắc bằng giọng nói tiếng Việt tự nhiên theo ngữ cảnh môi trường và cảm biến.
"""

from typing import List, Optional
from pydantic import BaseModel
from app.vision.decision_engine import get_vietnamese_name


class AssistantAskRequest(BaseModel):
    question: str
    current_step: Optional[str] = "Chặng 1: Đi thẳng 25m"
    detected_objects: Optional[List[str]] = []
    is_safe: Optional[bool] = True
    closest_distance: Optional[float] = 2.5


def generate_assistant_answer(
    question: str,
    current_step: str = "Chặng 1: Đi thẳng 25m",
    detected_objects: Optional[List[str]] = None,
    is_safe: bool = True,
    closest_distance: Optional[float] = 2.5,
) -> str:
    """
    Xử lý câu hỏi của người dùng và sinh câu trả lời súc tích, an toàn, dễ hiểu.
    """
    q = (question or "").strip().lower()
    raw_objs = detected_objects or []
    vn_objs = [get_vietnamese_name(x) for x in raw_objs]

    # 1. HỎI VỀ AN TOÀN / VẬT CẢN PHÍA TRƯỚC
    if any(k in q for k in ["an toàn", "vật cản", "chướng ngại", "đi được không", "thông thoáng", "phía trước", "có gì", "nguy hiểm"]):
        if not is_safe or (closest_distance is not None and closest_distance < 1.8):
            obs_str = ", ".join(vn_objs[:2]) if vn_objs else "vật cản"
            dist_val = f"{closest_distance:.1f}" if closest_distance is not None else "1"
            return f"Cảnh báo! Lối đi phía trước có {obs_str}, cách khoảng {dist_val} mét. Bạn hãy đi chậm lại hoặc chú ý bước chân."
        else:
            return "Lối đi phía trước hiện rất thông thoáng và an toàn. Bạn có thể tự tin tiếp tục bước đi thẳng."

    # 2. HỎI VỀ CON NGƯỜI
    elif any(k in q for k in ["người", "ai đó", "có ai"]):
        has_person = any("person" in str(x).lower() or "người" in str(x).lower() for x in raw_objs)
        if has_person:
            dist_str = f"cách khoảng {closest_distance:.1f} mét" if closest_distance else "ở ngay trước mặt"
            return f"Camera phát hiện có người ở phía trước, {dist_str}. Bạn hãy giữ khoảng cách an toàn."
        else:
            return "Hiện tại không phát hiện người nào trong lối đi trước mặt bạn."

    # 3. HỎI VỊ TRÍ HIỆN TẠI
    elif any(k in q for k in ["ở đâu", "vị trí", "điểm đến", "đến đâu", "tôi đang ở đâu", "tọa độ"]):
        return f"Bạn đang ở {current_step}. Hãy chú ý quan sát xung quanh và tiếp tục lộ trình."

    # 4. HỎI HƯỚNG ĐI / BƯỚC TIẾP THEO
    elif any(k in q for k in ["làm gì", "tiếp theo", "hướng dẫn", "rẽ", "đi đâu", "hướng nào", "bước tiếp"]):
        return f"Bước tiếp theo: Bạn hãy {current_step}. Chú ý giữ thẳng hướng di chuyển."

    # 5. HỎI VỀ ĐỌC CHỮ / BIỂN BÁO
    elif any(k in q for k in ["đọc chữ", "biển báo", "chữ gì", "đọc biển", "đọc sách", "tài liệu", "chữ"]):
        return "Chức năng đọc chữ sẵn sàng. Bạn hãy hướng camera về phía biển báo hoặc văn bản, tôi sẽ đọc to nội dung cho bạn."

    # 6. YÊU CẦU TRỢ GIÚP / HƯỚNG DẪN CÂU HỎI
    elif any(k in q for k in ["giúp", "trợ giúp", "hướng dẫn tôi", "bạn làm được gì", "câu lệnh", "hỏi gì"]):
        return "Tôi có thể hỗ trợ bạn: Kiểm tra an toàn phía trước, phát hiện người, đọc biển báo, hoặc thông báo vị trí hiện tại."

    # 7. CHÀO HỎI / XÃ GIAO
    elif any(k in q for k in ["chào", "xin chào", "bạn là ai", "tên gì"]):
        return "Xin chào bạn! Tôi là trợ lý thị giác AI. Tôi luôn đồng hành quan sát và hỗ trợ bạn di chuyển an toàn."

    # 8. MẶC ĐỊNH SÚC TÍCH, DỄ HIỂU
    else:
        return "Tôi đang lắng nghe. Bạn có thể hỏi: Phía trước an toàn không, có người không, hoặc tôi đang ở đâu."
