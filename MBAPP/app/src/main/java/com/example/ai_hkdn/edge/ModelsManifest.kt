package com.example.ai_hkdn.edge

import com.google.gson.annotations.SerializedName

/**
 * Cấu hình manifest sinh tự động từ `backend/app/vision/npu_exporter.py`
 * giúp Kotlin đồng bộ 100% nhãn lớp và kích thước tensor của 4 mô hình ONNX FP16.
 */
data class ModelsManifest(
    @SerializedName("schema_version") val schemaVersion: Int = 1,
    @SerializedName("target_hardware") val targetHardware: String = "",
    @SerializedName("yolo_imgsz") val yoloImgsz: Int = 320,
    @SerializedName("depth_imgsz") val depthImgsz: Int = 252,
    @SerializedName("seg_imgsz") val segImgsz: Int = 256,
    @SerializedName("imagenet_mean") val imagenetMean: List<Float> = listOf(0.485f, 0.456f, 0.406f),
    @SerializedName("imagenet_std") val imagenetStd: List<Float> = listOf(0.229f, 0.224f, 0.225f),
    @SerializedName("yoloe_names") val yoloeNames: List<String> = emptyList(),
    @SerializedName("obstacle_names") val obstacleNames: List<String> = emptyList(),
    @SerializedName("segformer_id2label") val segformerId2Label: Map<String, String> = emptyMap(),
    @SerializedName("models") val models: Map<String, EdgeModelFileMeta> = emptyMap(),
)

data class EdgeModelFileMeta(
    @SerializedName("filename") val filename: String,
    @SerializedName("size_bytes") val sizeBytes: Long = 0L,
    @SerializedName("size_mb") val sizeMb: Double = 0.0,
    @SerializedName("sha256") val sha256: String = "",
)
