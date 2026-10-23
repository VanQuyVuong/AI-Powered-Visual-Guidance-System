"""
Entrypoint thuận tiện: Cho phép chạy `python main.py` từ thư mục gốc của dự án.
"""
import os
import sys

# Thêm thư mục backend vào sys.path
backend_dir = os.path.join(os.path.dirname(__file__), "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import uvicorn

if __name__ == "__main__":
    print("🚀 Khởi động AI-Powered Visual Guidance System...")
    # Chạy server uvicorn từ backend.main:app
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
