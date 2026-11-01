package com.example.ai_hkdn.data.model

import com.google.gson.annotations.SerializedName

/**
 * Phản hồi phân tích thị giác từ AI Backend WebSocket
 */
data class VisionResponse(
    @SerializedName("success") val success: Boolean,
    @SerializedName("mode") val mode: String? = null,
    @SerializedName("objects") val objects: List<DetectedObject>? = null,
    @SerializedName("guidance") val guidance: GuidanceInfo? = null,
    @SerializedName("navigation") val navigation: NavigationInfo? = null,
    @SerializedName("error") val error: String? = null
)

/**
 * Chỉ dẫn điều hướng 3 làn (Dẫn đường thời gian thực)
 */
data class NavigationInfo(
    @SerializedName("direction") val direction: String = "straight", // straight, left, right, stop
    @SerializedName("confidence") val confidence: Float = 0.9f,
    @SerializedName("text") val text: String? = null
)

/**
 * Vật thể được nhận diện và bám đuổi (YOLO + ByteTrack + ROI)
 */
data class DetectedObject(
    @SerializedName("id") val id: Int? = null,
    @SerializedName("class_name") val className: String,
    @SerializedName("class_name_vn") val classNameVn: String? = null,
    @SerializedName("confidence") val confidence: Float,
    @SerializedName("position") val position: String = "center", // left, center, right
    @SerializedName("distance_m") val distanceM: Float? = null,
    @SerializedName("is_danger") val isDanger: Boolean? = null,
    @SerializedName("bounding_box") val boundingBox: BoundingBox
)

/**
 * Tọa độ hộp bao quanh vật thể
 */
data class BoundingBox(
    @SerializedName("x1") val x1: Int,
    @SerializedName("y1") val y1: Int,
    @SerializedName("x2") val x2: Int,
    @SerializedName("y2") val y2: Int
) {
    val width: Int get() = x2 - x1
    val height: Int get() = y2 - y1
    val centerX: Float get() = (x1 + x2) / 2f
    val bottomY: Float get() = y2.toFloat()
}

/**
 * Thông tin cảnh báo và chỉ dẫn bằng giọng nói
 */
data class GuidanceInfo(
    @SerializedName("text") val text: String,
    @SerializedName("priority") val priority: String = "LOW" // LOW, MEDIUM, HIGH, DANGER
)
