"""
Entrypoint thuận tiện: Cho phép chạy `python main.py` trực tiếp từ thư mục gốc của dự án.
Tự động chuyển sang môi trường ảo `.venv` (có đủ PyTorch, Transformers, Depth Anything V2, SAM),
tự động phát hiện IP Wi-Fi, in bảng hướng dẫn và tự động mở trình duyệt web.
"""
import os
import sys
import subprocess

# 1. Tự động chuyển sang Python trong .venv nếu người dùng chạy bằng Python hệ thống/Conda
_root_dir = os.path.dirname(os.path.abspath(__file__))
_venv_python = os.path.join(_root_dir, ".venv", "Scripts", "python.exe")
if os.path.exists(_venv_python):
    if os.path.normcase(os.path.abspath(sys.executable)) != os.path.normcase(os.path.abspath(_venv_python)):
        print(f"🔄 Tự động chuyển sang môi trường AI (.venv): {_venv_python}")
        sys.exit(subprocess.call([_venv_python, "-u"] + sys.argv))

# 2. Đảm bảo in tiếng Việt chuẩn UTF-8 trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# 3. Thêm thư mục backend vào sys.path và chuyển cwd về backend
backend_dir = os.path.join(_root_dir, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
os.chdir(backend_dir)

import threading
import webbrowser
import time
import uvicorn
from main import app, get_local_ip

if __name__ == "__main__":
    local_ip = get_local_ip()

    print("\n" + "=" * 70)
    print("🚀 AI VISUAL GUIDANCE SYSTEM - ĐẦU NÃO XỬ LÝ ĐÃ SẴN SÀNG!")
    print(f"🌐 Màn hình Web Monitor (Laptop):     http://localhost:8000/view-phone")
    print(f"📱 ĐỊA CHỈ NHẬP TRÊN APP ĐIỆN THOẠI:  {local_ip}:8000")
    print(f"📥 Link tải APK mới nhất qua Wi-Fi:   http://{local_ip}:8000/download-apk")
    print("=" * 70)
    print("🌐 Đang tự động mở màn hình Web Monitor trên trình duyệt...\n")

    def _open_browser():
        time.sleep(2.0)
        try:
            webbrowser.open("http://localhost:8000/view-phone")
        except Exception:
            pass

    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
