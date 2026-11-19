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
    useEdgeBitmap: Boolean = true,
    canProcessFrame: () -> Boolean = { true },
    onFrameCaptured: (ByteArray, String) -> Unit,
    onEdgeBitmapCaptured: (android.graphics.Bitmap) -> Unit = {},
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    val currentUseEdgeBitmap by rememberUpdatedState(useEdgeBitmap)
    val currentCanProcessFrame by rememberUpdatedState(canProcessFrame)
    val currentOnFrameCaptured by rememberUpdatedState(onFrameCaptured)
    val currentOnEdgeBitmapCaptured by rememberUpdatedState(onEdgeBitmapCaptured)
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            cameraExecutor.shutdown()
        }
    }

    LaunchedEffect(isStreaming, useEdgeBitmap) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases(
                context = context,
                lifecycleOwner = lifecycleOwner,
                cameraProvider = cameraProvider,
                previewView = previewView,
                isStreaming = isStreaming,
                useEdgeBitmap = { currentUseEdgeBitmap },
                canProcessFrame = { currentCanProcessFrame() },
                executor = cameraExecutor,
                onFrameCaptured = { bytes, b64 -> currentOnFrameCaptured(bytes, b64) },
                onEdgeBitmapCaptured = { bmp -> currentOnEdgeBitmapCaptured(bmp) }
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
    useEdgeBitmap: () -> Boolean,
    canProcessFrame: () -> Boolean,
    executor: java.util.concurrent.ExecutorService,
    onFrameCaptured: (ByteArray, String) -> Unit,
    onEdgeBitmapCaptured: (android.graphics.Bitmap) -> Unit
) {
    val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
    val displayRotation = previewView.display?.rotation ?: android.view.Surface.ROTATION_0
    val targetRatio = androidx.camera.core.AspectRatio.RATIO_4_3

    val preview = Preview.Builder()
        .setTargetAspectRatio(targetRatio)
        .setTargetRotation(displayRotation)
        .build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }

    // Khóa độ phân giải tối ưu 480x640 (tỷ lệ 4:3) và chỉ trích xuất khung hình khi bộ xử lý sẵn sàng
    val imageAnalysis = ImageAnalysis.Builder()
        .setTargetResolution(Size(480, 640))
        .setTargetRotation(displayRotation)
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
        .build()

    imageAnalysis.setAnalyzer(executor) { imageProxy: ImageProxy ->
        try {
            if (isStreaming && canProcessFrame()) {
                if (useEdgeBitmap()) {
                    val bmp = ImageUtils.imageProxyToRotatedBitmap(imageProxy, maxSide = 640)
                    if (bmp != null) {
                        onEdgeBitmapCaptured(bmp)
                    }
                } else {
                    val jpegBytes = ImageUtils.imageProxyToJpegBytes(imageProxy, quality = 68)
                    if (jpegBytes != null) {
                        val base64 = ImageUtils.jpegBytesToBase64(jpegBytes)
                        onFrameCaptured(jpegBytes, base64)
                    }
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
