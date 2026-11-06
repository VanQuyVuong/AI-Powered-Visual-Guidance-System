"""
File khởi động nhanh toàn bộ hệ thống AI Visual Guidance System.
Người dùng chỉ cần mở terminal tại thư mục gốc dự án và chạy:
    python run.py
"""

import os
import sys
import subprocess

# Đảm bảo in tiếng Việt chuẩn trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(root_dir, "backend")

    # Ưu tiên dùng python trong môi trường ảo .venv nếu có
    venv_python = os.path.join(root_dir, ".venv", "Scripts", "python.exe")
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable

    main_script = os.path.join(backend_dir, "main.py")

    os.environ["PYTHONPATH"] = backend_dir

    print(f"Khởi động hệ thống với Python: {python_exe}")
    cmd = [python_exe, "-u", main_script]
    try:
        subprocess.run(cmd, cwd=backend_dir)
    except KeyboardInterrupt:
        print("\nĐã dừng hệ thống.")

if __name__ == "__main__":
    main()
