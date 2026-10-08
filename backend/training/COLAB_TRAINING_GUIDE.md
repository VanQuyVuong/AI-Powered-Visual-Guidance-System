# Hướng dẫn Fine-tune YOLOv8 trên Google Colab

Do máy tính của bạn cấu hình yếu và chưa có sẵn dữ liệu, cách tốt nhất, nhanh nhất và hoàn toàn miễn phí là sử dụng **Google Colab (chạy GPU đám mây)** kết hợp với **Dữ liệu cộng đồng từ Roboflow**.

Bạn chỉ cần làm theo các bước sao chép / dán (Copy/Paste) cực kỳ đơn giản dưới đây.

## Bước 1: Chuẩn bị Môi trường trên Google Colab
1. Truy cập vào [Google Colab](https://colab.research.google.com/) và đăng nhập bằng tài khoản Google.
2. Bấm vào **New Notebook** (Sổ tay mới).
3. Ở menu phía trên, chọn **Runtime** -> **Change runtime type** (Thay đổi loại thời gian chạy) -> Chọn phần cứng là **T4 GPU** và bấm Save. (Điều này giúp train AI nhanh gấp 10 lần máy thường).

## Bước 2: Cài đặt thư viện (Paste vào Ô code 1)
Dán đoạn mã sau vào ô code đầu tiên và bấm nút Play (▶) để chạy:

```python
!pip install ultralytics roboflow
```

## Bước 3: Tải Dataset miễn phí từ Roboflow (Paste vào Ô code 2)
Roboflow Universe có hàng ngàn bộ dữ liệu miễn phí. Ở đây chúng ta sẽ lấy bộ dữ liệu "Pothole" (Ổ gà) làm ví dụ. Bạn có thể tìm thêm các bộ dữ liệu khác (như nắp cống, cột điện) trên [Roboflow Universe](https://universe.roboflow.com/) và thay API key.

```python
from roboflow import Roboflow

# Khởi tạo thư mục datasets
import os
os.makedirs('/content/datasets', exist_ok=True)
%cd /content/datasets

# (Lưu ý: Đoạn mã dưới đây là mẫu công khai tải tập dữ liệu Ổ Gà - Potholes)
# Khi bạn đăng ký tài khoản miễn phí trên Roboflow, họ sẽ cấp cho bạn đoạn mã riêng với API_KEY của bạn.
rf = Roboflow(api_key="BẠN_ĐĂNG_KÝ_ROBOFLOW_ĐỂ_LẤY_API_KEY_FREE")
project = rf.workspace("viren-dhanwani").project("pothole-detection-1")
version = project.version(2)
dataset = version.download("yolov8")

print("Đã tải xong Dataset vào: ", dataset.location)
```
*(Mẹo: Việc đăng ký tài khoản Roboflow chỉ mất 30 giây bằng tài khoản Google).*

## Bước 4: Bắt đầu Huấn luyện AI (Paste vào Ô code 3)
Tiếp tục tạo ô code mới và chạy lệnh huấn luyện. Lệnh này sẽ tải mô hình `yolov8n.pt` gốc về và huấn luyện thêm với dữ liệu vừa tải.

```python
from ultralytics import YOLO

# Tải mô hình YOLOv8 bản nhẹ nhất (Nano)
model = YOLO('yolov8n.pt') 

# Bắt đầu huấn luyện
# Thay đường dẫn file data.yaml bằng đường dẫn thực tế in ra từ bước 3
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,       # Số vòng học (có thể tăng lên 100 nếu cần)
    imgsz=640,       # Kích thước ảnh
    batch=16,        # Cụm dữ liệu mỗi lần học
    project='guidance_system',
    name='pothole_model'
)
```

## Bước 5: Lấy mô hình về máy (Paste vào Ô code 4)
Sau khi quá trình học xong (tùy số vòng học, mất khoảng 15-30 phút), file mô hình trí tuệ nhân tạo sẽ được lưu lại dưới tên `best.pt`. Chúng ta sẽ tải nó về máy tính:

```python
from google.colab import files

# Đường dẫn file mô hình sau khi train xong
model_path = '/content/datasets/guidance_system/pothole_model/weights/best.pt'

# Tải file về máy tính
files.download(model_path)
```

---

## Bước 6: Thay thế file trong dự án
Sau khi file `best.pt` được tải về máy của bạn:
1. Bạn đổi tên nó thành `guidance_model.pt`.
2. Bỏ file đó vào thư mục gốc của dự án `AI-Powered-Visual-Guidance-System`.
3. Trong các code nhận diện hiện tại, hãy đổi đường dẫn từ `yolov8n.pt` thành `guidance_model.pt`. AI của bạn giờ đã biết nhận diện các chướng ngại vật thực tế!
