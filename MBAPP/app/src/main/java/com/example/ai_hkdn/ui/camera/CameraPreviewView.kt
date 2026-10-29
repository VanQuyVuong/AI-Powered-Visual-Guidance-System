package com.example.ai_hkdn.ui.camera

import android.content.Context
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.example.ai_hkdn.util.ImageUtils
import java.util.concurrent.Executors

/**
 * Component CameraX thời gian thực:
 * - Khóa độ phân giải tối ưu 640x480 (kích thước chuẩn của mạng YOLO)
 * - Xử lý ảnh trực tiếp trong RAM (Zero-Disk I/O) giúp giảm độ trễ tối đa
 */
@Composable
fun CameraPreviewView(
    isStreaming: Boolean,
    onFrameCaptured: (ByteArray, String) -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
    val previewView = remember { PreviewView(context) }

    DisposableEffect(Unit) {
        onDispose {
            cameraExecutor.shutdown()
        }
    }

    LaunchedEffect(isStreaming) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases(
                context = context,
                lifecycleOwner = lifecycleOwner,
                cameraProvider = cameraProvider,
                previewView = previewView,
                isStreaming = isStreaming,
                executor = cameraExecutor,
                onFrameCaptured = onFrameCaptured
            )
        }, ContextCompat.getMainExecutor(context))
    }

    AndroidView(
        factory = { previewView },
        modifier = modifier.fillMaxSize()
    )
}

private fun bindCameraUseCases(
    context: Context,
    lifecycleOwner: LifecycleOwner,
    cameraProvider: ProcessCameraProvider,
    previewView: PreviewView,
    isStreaming: Boolean,
    executor: java.util.concurrent.ExecutorService,
    onFrameCaptured: (ByteArray, String) -> Unit
) {
    val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

    val preview = Preview.Builder().build().also {
        it.setSurfaceProvider(previewView.surfaceProvider)
    }

    // Cấu hình ImageAnalysis ở độ phân giải 640x480 (chuẩn YOLO)
    // STRATEGY_KEEP_ONLY_LATEST: Tự động bỏ qua các frame cũ nếu máy chưa kịp xử lý
    val imageAnalysis = ImageAnalysis.Builder()
        .setTargetResolution(Size(640, 480))
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
        .build()

    imageAnalysis.setAnalyzer(executor) { imageProxy: ImageProxy ->
        try {
            if (isStreaming) {
                val jpegBytes = ImageUtils.imageProxyToJpegBytes(imageProxy, quality = 70)
                if (jpegBytes != null) {
                    val base64 = ImageUtils.jpegBytesToBase64(jpegBytes)
                    onFrameCaptured(jpegBytes, base64)
                }
            }
        } catch (e: Exception) {
            Log.e("CameraPreviewView", "Lỗi phân tích frame: ${e.message}")
        } finally {
            imageProxy.close() // BẮT BUỘC close để nhả buffer cho frame tiếp theo
        }
    }

    try {
        cameraProvider.unbindAll()
        cameraProvider.bindToLifecycle(
            lifecycleOwner,
            cameraSelector,
            preview,
            imageAnalysis
        )
    } catch (e: Exception) {
        Log.e("CameraPreviewView", "Lỗi gắn Camera UseCases: ${e.message}")
    }
}
