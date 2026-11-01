package com.example.ai_hkdn.ui.hud

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import com.example.ai_hkdn.data.model.DetectedObject

/**
 * Lớp phủ AR HUD (Augmented Reality Heads-Up Display)
 * - Vẽ Vùng An Toàn ROI (Hình thang 3D mô phỏng làn đường cho người khiếm thị)
 * - Vẽ Bounding Box quanh các vật cản được AI nhận diện theo thời gian thực
 */
@Composable
fun ARHudOverlay(
    objects: List<DetectedObject>,
    imageSourceWidth: Int = 480, // Chiều rộng frame sau khi xoay dọc
    imageSourceHeight: Int = 640, // Chiều cao frame sau khi xoay dọc
    showRoi: Boolean = true,
    modifier: Modifier = Modifier
) {
    Canvas(modifier = modifier.fillMaxSize()) {
        val screenW = size.width
        val screenH = size.height

        // 1. Vẽ Vùng An Toàn ROI (Hình thang 3D phối cảnh lối đi)
        if (showRoi) {
            drawPerspectiveRoi(screenW, screenH)
        }

        // Khớp 1:1 với chế độ PreviewView.ScaleType.FILL_CENTER của camera:
        // Phóng to đồng dạng theo cạnh lớn nhất và căn giữa khung hình (tránh méo tỷ lệ)
        val srcW = imageSourceWidth.coerceAtLeast(1).toFloat()
        val srcH = imageSourceHeight.coerceAtLeast(1).toFloat()
        val scale = maxOf(screenW / srcW, screenH / srcH)
        val offsetX = (screenW - srcW * scale) / 2f
        val offsetY = (screenH - srcH * scale) / 2f

        // 2. Vẽ các vật thể phát hiện được
        objects.forEach { obj ->
            val box = obj.boundingBox
            val left = box.x1 * scale + offsetX
            val top = box.y1 * scale + offsetY
            val right = box.x2 * scale + offsetX
            val bottom = box.y2 * scale + offsetY
            val width = right - left
            val height = bottom - top

            // Xác định xem vật thể có nguy hiểm không (từ cờ is_danger của server, ở vị trí center hoặc gần)
            val isDangerous = (obj.isDanger == true) || obj.position.equals("center", ignoreCase = true) || (height / screenH > 0.40f)
            val boxColor = if (isDangerous) Color(0xFFFF2A55) else Color(0xFF00F0FF)

            // Vẽ hộp viền
            drawRect(
                color = boxColor,
                topLeft = Offset(left, top),
                size = Size(width, height),
                style = Stroke(width = 5f)
            )

            // Vẽ góc định vị HUD Cyberpunk
            val cornerLen = 24f
            // Góc trên-trái
            drawLine(boxColor, Offset(left, top), Offset(left + cornerLen, top), 7f)
            drawLine(boxColor, Offset(left, top), Offset(left, top + cornerLen), 7f)
            // Góc dưới-phải
            drawLine(boxColor, Offset(right, bottom), Offset(right - cornerLen, bottom), 7f)
            drawLine(boxColor, Offset(right, bottom), Offset(right, bottom - cornerLen), 7f)

            // Vẽ nhãn đối tượng (tên tiếng Việt + khoảng cách mét)
            drawContext.canvas.nativeCanvas.apply {
                val paint = android.graphics.Paint().apply {
                    color = android.graphics.Color.WHITE
                    textSize = 34f
                    isFakeBoldText = true
                    setShadowLayer(8f, 0f, 0f, android.graphics.Color.BLACK)
                }

                val bgPaint = android.graphics.Paint().apply {
                    color = if (isDangerous) android.graphics.Color.argb(220, 220, 20, 60)
                    else android.graphics.Color.argb(200, 0, 150, 180)
                }

                val displayName = if (!obj.classNameVn.isNullOrBlank()) obj.classNameVn else obj.className
                val distInfo = if (obj.distanceM != null && obj.distanceM > 0f) {
                    String.format(java.util.Locale.US, "%.1fm", obj.distanceM)
                } else {
                    "${(obj.confidence * 100).toInt()}%"
                }
                val label = "$displayName $distInfo"
                val textBounds = android.graphics.Rect()
                paint.getTextBounds(label, 0, label.length, textBounds)

                val tagLeft = left.coerceAtLeast(10f)
                val tagTop = (top - 12f).coerceAtLeast(textBounds.height().toFloat() + 20f)

                drawRoundRect(
                    tagLeft - 8f,
                    tagTop - textBounds.height() - 10f,
                    tagLeft + textBounds.width() + 16f,
                    tagTop + 8f,
                    8f, 8f,
                    bgPaint
                )
                drawText(label, tagLeft, tagTop, paint)
            }
        }
    }
}

/**
 * Vẽ làn đường hình thang 3D an toàn (Region of Interest)
 */
private fun DrawScope.drawPerspectiveRoi(w: Float, h: Float) {
    val topY = h * 0.45f
    val botY = h * 0.98f

    val topWidth = w * 0.35f
    val botWidth = w * 0.78f

    val topStartX = (w - topWidth) / 2f
    val topEndX = topStartX + topWidth

    val botStartX = (w - botWidth) / 2f
    val botEndX = botStartX + botWidth

    val path = Path().apply {
        moveTo(topStartX, topY)
        lineTo(topEndX, topY)
        lineTo(botEndX, botY)
        lineTo(botStartX, botY)
        close()
    }

    // Tô màu mờ làn đường an toàn
    drawPath(
        path = path,
        color = Color(0x2200FF88)
    )

    // Viền xanh neon phát sáng
    drawPath(
        path = path,
        color = Color(0xFF00FF88),
        style = Stroke(width = 3.5f)
    )

    // Đường tâm dẫn hướng ở giữa lối đi
    drawLine(
        color = Color(0x8800FF88),
        start = Offset(w / 2f, topY),
        end = Offset(w / 2f, botY),
        strokeWidth = 2.5f
    )
}
