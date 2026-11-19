"""
Model Scheduler - Bộ điều phối tài nguyên CPU và chạy lệch pha cho 4 mô hình AI.

Mục đích:
- Cân bằng số luồng PyTorch trên CPU (1/2 số logical cores, tối đa 6) để luồng chính YOLO
  và 1 luồng nền bổ trợ cùng tận dụng CPU mà không tranh chấp cache/context-switch.
- Đảm bảo Depth Anything V2 / SegFormer và EdgeSAM chạy lệch pha nhau (mutual exclusion)
  thông qua khóa atomic, loại bỏ phụ thuộc vòng tròn giữa các service.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

_THREADS_CONFIGURED = False
_CONFIG_LOCK = threading.Lock()


def configure_cpu_threads() -> int:
    """Thiết lập số luồng PyTorch tối ưu trên CPU (trả về số luồng hiện hành, hoặc 0 nếu dùng CUDA)."""
    global _THREADS_CONFIGURED
    try:
        import torch
    except Exception:
        return 0

    if torch.cuda.is_available():
        return 0

    target_threads = min(6, max(2, (os.cpu_count() or 4) // 2))
    with _CONFIG_LOCK:
        if not _THREADS_CONFIGURED or torch.get_num_threads() != target_threads:
            torch.set_num_threads(target_threads)
            _THREADS_CONFIGURED = True
    return target_threads


class AuxModelCoordinator:
    """Điều phối chạy nền lệch pha giữa DepthService và SamService để tránh nghẽn CPU."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_owner: Optional[str] = None

    def try_acquire(self, owner: str) -> bool:
        """Giành quyền chạy nền nếu chưa có model bổ trợ nào đang chiếm CPU."""
        with self._lock:
            if self._active_owner is not None:
                return False
            self._active_owner = owner
            return True

    def release(self, owner: str) -> None:
        """Trả quyền chạy nền khi worker hoàn tất."""
        with self._lock:
            if self._active_owner == owner:
                self._active_owner = None

    def current_owner(self) -> Optional[str]:
        """Tên model nền đang chạy (hoặc None nếu đang rảnh)."""
        with self._lock:
            return self._active_owner


aux_coordinator = AuxModelCoordinator()
