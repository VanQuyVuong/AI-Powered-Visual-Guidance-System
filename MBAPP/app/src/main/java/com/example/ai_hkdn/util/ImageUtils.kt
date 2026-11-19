package com.example.ai_hkdn.util

import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Base64
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

object ImageUtils {

    /**
     * Chuyển đổi ImageProxy trực tiếp từ RAM thành Bitmap xoay đúng chiều (Zero-Disk I/O)
     * phục vụ suy luận NPU/ONNX trực tiếp trên thiết bị không tốn chi phí nén JPEG.
     */
    @OptIn(ExperimentalGetImage::class)
    fun imageProxyToRotatedBitmap(imageProxy: ImageProxy, maxSide: Int = 640): Bitmap? {
        return try {
            val bitmap = imageProxy.toBitmap()
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees
            val rotatedBitmap = if (rotationDegrees != 0) {
                val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true).also {
                    if (it != bitmap) bitmap.recycle()
                }
            } else {
                bitmap
            }

            val longest = maxOf(rotatedBitmap.width, rotatedBitmap.height)
            if (longest > maxSide) {
                val scale = maxSide.toFloat() / longest.toFloat()
                val nw = (rotatedBitmap.width * scale).toInt().coerceAtLeast(32)
                val nh = (rotatedBitmap.height * scale).toInt().coerceAtLeast(32)
                Bitmap.createScaledBitmap(rotatedBitmap, nw, nh, true).also {
                    if (it != rotatedBitmap) rotatedBitmap.recycle()
                }
            } else {
                rotatedBitmap
            }
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Chuyển đổi ImageProxy trực tiếp từ RAM thành mảng byte JPEG
     * - Xoay đúng chiều theo cảm biến máy ảnh (rotationDegrees)
     * - Nén JPEG ở chất lượng 70% để giảm kích thước xuống ~30KB (truyền mạng <10ms)
     */
    @OptIn(ExperimentalGetImage::class)
    fun imageProxyToJpegBytes(imageProxy: ImageProxy, quality: Int = 68, maxSide: Int = 640): ByteArray? {
        return try {
            val finalBitmap = imageProxyToRotatedBitmap(imageProxy, maxSide = maxSide) ?: return null
            val outputStream = ByteArrayOutputStream()
            finalBitmap.compress(Bitmap.CompressFormat.JPEG, quality, outputStream)
            finalBitmap.recycle()
            outputStream.toByteArray()
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Chuyển ByteArray JPEG thành chuỗi Base64 gọn gàng để truyền qua WebSocket
     */
    fun jpegBytesToBase64(jpegBytes: ByteArray): String {
        return Base64.encodeToString(jpegBytes, Base64.NO_WRAP)
    }
}
