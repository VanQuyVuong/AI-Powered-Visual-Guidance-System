"""Kiểm thử cho module xuất mô hình ONNX FP16 (NPU/Edge) và luồng WebSocket EDGE_TELEMETRY."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


def test_npu_exporter_mediatek_plan_and_manifest_schema():
    from app.vision.npu_exporter import DEFAULT_ANDROID_MODELS_DIR, build_models_manifest, npu_exporter

    plan = npu_exporter.get_npu_deployment_plan("mediatek_dimensity")
    assert "Dimensity 7300-Ultra" in plan["target_hardware"]["chipset"]
    assert plan["target_hardware"]["recommended_precision"] == "FP16"
    assert len(plan["models"]) == 4
    assert plan["compression_ratio"] >= 1.5

    manifest = build_models_manifest(
        yoloe_names=["person", "stairs"],
        obstacle_names=["pothole"],
        segformer_id2label={"0": "wall", "53": "stairs"},
        model_files={},
        yolo_imgsz=320,
        depth_imgsz=252,
        seg_imgsz=256,
    )
    assert manifest["schema_version"] == 1
    assert manifest["yolo_imgsz"] == 320
    assert manifest["depth_imgsz"] == 252
    assert manifest["seg_imgsz"] == 256
    assert manifest["yoloe_names"] == ["person", "stairs"]

    live_manifest_path = DEFAULT_ANDROID_MODELS_DIR / "models_manifest.json"
    if live_manifest_path.exists():
        live_data = json.loads(live_manifest_path.read_text(encoding="utf-8"))
        assert len(live_data["yoloe_names"]) == 37
        assert len(live_data["obstacle_names"]) == 8
        assert len(live_data["segformer_id2label"]) == 150
        for key in ("yoloe", "obstacle", "depth", "segformer"):
            assert key in live_data["models"]


def test_websocket_edge_telemetry_mirrors_to_viewers_without_running_server_ai():
    import main as backend_main

    client = TestClient(backend_main.app)
    sample_telemetry = {
        "type": "EDGE_TELEMETRY",
        "mode": "EDGE_ON_DEVICE_NPU",
        "device": "Redmi Note 14 Pro 5G",
        "processing_ms": 46.5,
        "objects": [
            {
                "class_name": "stairs",
                "class_name_vn": "cầu thang",
                "confidence": 0.89,
                "position": "center",
                "distance_m": 1.6,
                "in_roi": True,
                "is_danger": True,
                "bounding_box": {"x1": 90, "y1": 240, "x2": 390, "y2": 580},
            }
        ],
        "guidance": {
            "text": "Chú ý dưới chân, có cầu thang, hãy bước chậm.",
            "display_text": "Chú ý dưới chân, có cầu thang, hãy bước chậm.",
            "priority": "HIGH",
        },
        "navigation": {"direction": "stop", "text": "Bước chậm"},
        "img_size": {"width": 480, "height": 640},
        "frame_b64": "AQID",
    }

    with patch.object(backend_main, "analyze_frame_sync") as mock_analyze:
        with client.websocket_connect("/api/v1/vision/stream") as viewer_ws:
            viewer_ws.send_text("VIEW_ONLY")
            with client.websocket_connect("/api/v1/vision/stream") as phone_ws:
                phone_ws.send_text(json.dumps(sample_telemetry))
                relayed = viewer_ws.receive_json()

        mock_analyze.assert_not_called()
        assert relayed["success"] is True
        assert relayed["mode"] == "EDGE_ON_DEVICE_NPU"
        assert relayed["device"] == "Redmi Note 14 Pro 5G"
        assert relayed["processing_ms"] == 46.5
        assert relayed["objects"][0]["class_name"] == "stairs"
        assert "cầu thang" in relayed["guidance"]["text"]
