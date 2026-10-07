# HƯỚNG DẪN HUẤN LUYỆN AI (TRAIN YOLOv8) DÀNH CHO TEAM DATA

Tài liệu này là "chìa khóa trao tay" giúp team AI dễ dàng huấn luyện một mô hình YOLOv8 độc quyền cho chiếc Mũ Thông Minh (Smart Hat). 

---

## BƯỚC 1: CHUẨN BỊ DỮ LIỆU (DATASET)
1. Hãy tạo một tài khoản miễn phí trên [Roboflow](https://roboflow.com/).
2. Tạo một Project mới dạng **Object Detection**.
3. Upload các hình ảnh/video bạn tự quay (hoặc tải trên mạng) về các chướng ngại vật ở Việt Nam. Khuyến nghị các nhãn (classes) sau:
   - `o_ga` (Ổ gà / vũng nước)
   - `nap_cong` (Nắp cống hở)
   - `vach_ke_duong` (Vạch sang đường / Zebra crossing)
4. Tự dùng chuột khoanh vùng (Label) các vật thể trong ảnh.
5. Sau khi Label xong khoảng 300 - 500 ảnh, ấn **Generate Dataset** và chọn **Export** (Chọn định dạng YOLOv8). Roboflow sẽ cấp cho bạn một đoạn code để tải data.

---

## BƯỚC 2: CHẠY GOOGLE COLAB (MÁY CHỦ GPU MIỄN PHÍ)
Máy tính cá nhân thường không đủ mạnh để Train AI, nên chúng ta sẽ dùng máy chủ của Google.
1. Truy cập [Google Colab](https://colab.research.google.com/) và tạo một Notebook mới.
2. Trên thanh menu, chọn **Runtime (Thời gian chạy)** -> **Change runtime type (Thay đổi loại thời gian chạy)**.
3. Ở mục **Hardware accelerator (Trình tăng tốc phần cứng)**, hãy chọn **T4 GPU** và lưu lại.

---

## BƯỚC 3: CÁC ĐOẠN CODE ĐỂ TRAIN (COPY & PASTE)

Vào Colab, tạo các ô Code (Cell) và dán lần lượt các đoạn lệnh sau rồi bấm nút Play (▶️) để chạy:

**Cell 1: Cài đặt công cụ**
```python
!pip install ultralytics roboflow
import ultralytics
ultralytics.checks() # Kiểm tra xem GPU đã nhận chưa
```

**Cell 2: Tải dữ liệu từ Roboflow**
*(Lưu ý: Thay đoạn code dưới đây bằng đoạn code mà Roboflow cấp cho bạn ở Bước 1)*
```python
from roboflow import Roboflow
rf = Roboflow(api_key="API_KEY_CUA_BAN_O_DAY")
project = rf.workspace("workspace_name").project("project_name")
version = project.version(1)
dataset = version.download("yolov8")
```

**Cell 3: Bắt đầu Huấn luyện (Training)**
Đây là lúc AI bắt đầu học. Quá trình này mất khoảng 1-2 tiếng tùy số lượng ảnh.
```python
from ultralytics import YOLO

# Tải bộ não cơ bản YOLOv8 Nano (nhẹ nhất, phù hợp chạy trên Mũ thông minh/Điện thoại)
model = YOLO('yolov8n.pt') 

# Bắt đầu dạy học (train)
# data: đường dẫn tới file data.yaml trong thư mục tải về từ Roboflow
# epochs: số lần học đi học lại (100 lần là đẹp)
# imgsz: kích thước ảnh (chuẩn 640)
results = model.train(data=f"{dataset.location}/data.yaml", epochs=100, imgsz=640)
```

**Cell 4: Lấy thành quả mang về**
Khi quá trình Train chạy đến 100%, file "bộ não" mới sẽ ra đời với tên gọi `best.pt`. Chạy lệnh này để tải nó về máy tính của bạn:
```python
from google.colab import files
# Tải file weights tốt nhất về máy
files.download('/content/runs/detect/train/weights/best.pt')
```

---
## BƯỚC 4: RÁP VÀO BACKEND CỦA CHÚNG TA
Sau khi team AI đưa cho bạn file `best.pt`, bạn chỉ cần:
1. Đổi tên nó thành `guidance_model_v1.pt`
2. Bỏ vào thư mục `backend/`
3. Mở file `backend/app/vision/yolo_service.py` và sửa dòng load model:
`model = YOLO('guidance_model_v1.pt')`
4. Xong! Mũ thông minh của bạn giờ đã biết né ổ gà Việt Nam!
