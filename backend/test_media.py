"""
Forwarder file duy trì tương thích: Chuyển tiếp tới backend/scripts/test_media.py
"""
import os
import runpy

if __name__ == "__main__":
    script_path = os.path.join(os.path.dirname(__file__), "scripts", "test_media.py")
    runpy.run_path(script_path, run_name="__main__")
