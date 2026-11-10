package com.example.ai_hkdn.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.example.ai_hkdn.data.model.DetectedObject
import com.example.ai_hkdn.service.*
import com.example.ai_hkdn.ui.camera.CameraPreviewView
import com.example.ai_hkdn.ui.hud.ARHudOverlay
import kotlinx.coroutines.launch

@Composable
fun SmartHatScreen() {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()

    // Quản lý quyền Camera
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        hasCameraPermission = isGranted
    }

    LaunchedEffect(Unit) {
        if (!hasCameraPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    if (!hasCameraPermission) {
        CameraPermissionDeniedContent {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
        return
    }

    // Các trạng thái của hệ thống
    var serverIp by remember { mutableStateOf("192.168.1.49:8000") }
    var isStreaming by remember { mutableStateOf(false) }
    var connectionStatus by remember { mutableStateOf(ConnectionStatus.DISCONNECTED) }
    var detectedObjects by remember { mutableStateOf<List<DetectedObject>>(emptyList()) }
    var guidanceText by remember { mutableStateOf("Sẵn sàng. Bấm nút Quét để bắt đầu điều hướng.") }
    var guidancePriority by remember { mutableStateOf("LOW") }
    var latencyMs by remember { mutableLongStateOf(0L) }
    var isTtsEnabled by remember { mutableStateOf(true) }
    var isOcrLoading by remember { mutableStateOf(false) }
    var showIpDialog by remember { mutableStateOf(false) }
    var latestJpegBytes by remember { mutableStateOf<ByteArray?>(null) }

    var srcFrameWidth by remember { mutableIntStateOf(480) }
    var srcFrameHeight by remember { mutableIntStateOf(640) }

    // Dịch vụ TTS & OCR & WebSocket
    val ttsManager = remember { TtsManager(context) }
    val ocrService = remember { OcrService("http://$serverIp") }

    val webSocketService = remember {
        WebSocketService(
            serverUrl = "ws://$serverIp/api/v1/vision/stream",
            onResult = { response, latency ->
                latencyMs = latency
                if (response.success) {
                    detectedObjects = response.objects ?: emptyList()
                    response.imgSize?.let { size ->
                        if (size.width > 0 && size.height > 0) {
                            srcFrameWidth = size.width
                            srcFrameHeight = size.height
                        }
                    }
                    val text = response.guidance?.text
                    val priority = response.guidance?.priority ?: "LOW"
                    if (!text.isNullOrBlank()) {
                        guidanceText = text
                        guidancePriority = priority
                        if (isTtsEnabled) {
                            ttsManager.speak(text, priority)
                        }
                    }
                }
            },
            onStatusChange = { newStatus ->
                connectionStatus = newStatus
                if (newStatus == ConnectionStatus.CONNECTED) {
                    guidanceText = "Đã kết nối AI Server! Đang phân tích camera..."
                    if (isTtsEnabled) ttsManager.speak("Đã kết nối hệ thống thị giác thông minh", force = true)
                } else if (newStatus == ConnectionStatus.DISCONNECTED && isStreaming) {
                    guidanceText = "⚠️ Mất kết nối máy chủ AI, hãy dừng lại!"
                    guidancePriority = "CRITICAL"
                    if (isTtsEnabled) ttsManager.speak("Chú ý, mất kết nối máy chủ, hãy dừng lại chờ tín hiệu.", priority = "CRITICAL", force = true)
                }
            },
            onError = { errMsg ->
                guidanceText = "⚠️ $errMsg"
                guidancePriority = "HIGH"
                if (isTtsEnabled && isStreaming) {
                    ttsManager.speak("Chú ý, kết nối mạng không ổn định, hãy bước chậm.", priority = "HIGH")
                }
            }
        )
    }

    DisposableEffect(Unit) {
        onDispose {
            webSocketService.disconnect()
            ttsManager.shutdown()
        }
    }

    // Cập nhật URL khi đổi IP hoặc dùng Ngrok domain
    LaunchedEffect(serverIp) {
        val cleanHost = serverIp.trim().removePrefix("http://").removePrefix("https://").removePrefix("ws://").removePrefix("wss://")
        val isSecure = serverIp.startsWith("https://") || serverIp.startsWith("wss://") || cleanHost.contains("ngrok") || cleanHost.contains("trycloudflare")
        val wsScheme = if (isSecure) "wss" else "ws"
        val httpScheme = if (isSecure) "https" else "http"
        
        webSocketService.updateServerUrl("$wsScheme://$cleanHost/api/v1/vision/stream")
        ocrService.updateServerUrl("$httpScheme://$cleanHost")
    }

    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        // 1. Tầng CameraX Preview
        CameraPreviewView(
            isStreaming = isStreaming,
            onFrameCaptured = { bytes, base64 ->
                latestJpegBytes = bytes
                if (isStreaming) {
                    webSocketService.sendFrame(base64)
                }
            },
            modifier = Modifier.fillMaxSize()
        )

        // 2. Tầng AR HUD Canvas (Vùng an toàn ROI & Bounding Box)
        ARHudOverlay(
            objects = detectedObjects,
            imageSourceWidth = srcFrameWidth,
            imageSourceHeight = srcFrameHeight,
            showRoi = isStreaming,
            modifier = Modifier.fillMaxSize()
        )

        // 3. Tầng Điều khiển & Bảng trạng thái trên cùng
        TopStatusBar(
            serverIp = serverIp,
            connectionStatus = connectionStatus,
            latencyMs = latencyMs,
            isTtsEnabled = isTtsEnabled,
            onToggleTts = {
                isTtsEnabled = !isTtsEnabled
                if (!isTtsEnabled) ttsManager.stop()
            },
            onOpenIpDialog = { showIpDialog = true },
            modifier = Modifier
                .align(Alignment.TopCenter)
                .fillMaxWidth()
                .statusBarsPadding()
                .padding(12.dp)
        )

        // 4. Tầng Chỉ dẫn & Nút bấm dưới cùng
        BottomControlPanel(
            guidanceText = guidanceText,
            priority = guidancePriority,
            isStreaming = isStreaming,
            isOcrLoading = isOcrLoading,
            onToggleStream = {
                if (isStreaming) {
                    webSocketService.disconnect()
                    ttsManager.stop()
                    detectedObjects = emptyList()
                    guidanceText = "Đã dừng quét camera."
                    isStreaming = false
                } else {
                    webSocketService.connect()
                    isStreaming = true
                }
            },
            onReadOcr = {
                val bytes = latestJpegBytes
                if (bytes != null) {
                    isOcrLoading = true
                    coroutineScope.launch {
                        ttsManager.speak("Đang đọc chữ trên biển báo...", force = true)
                        val result = ocrService.readTextFromImage(bytes)
                        isOcrLoading = false
                        result.onSuccess { text ->
                            guidanceText = "📖 Biển báo: $text"
                            ttsManager.speak(text, force = true)
                        }.onFailure { err ->
                            guidanceText = "Lỗi đọc chữ: ${err.message}"
                            ttsManager.speak("Không đọc được chữ.", force = true)
                        }
                    }
                } else {
                    guidanceText = "Chưa có hình ảnh để đọc chữ. Hãy bấm Quét trước."
                }
            },
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .navigationBarsPadding()
                .padding(16.dp)
        )

        // Hộp thoại cấu hình IP
        if (showIpDialog) {
            IpConfigDialog(
                currentIp = serverIp,
                onDismiss = { showIpDialog = false },
                onSave = { newIp ->
                    serverIp = newIp
                    showIpDialog = false
                }
            )
        }
    }
}

/**
 * Thanh trạng thái phía trên (HUD Top Bar)
 */
@Composable
private fun TopStatusBar(
    serverIp: String,
    connectionStatus: ConnectionStatus,
    latencyMs: Long,
    isTtsEnabled: Boolean,
    onToggleTts: () -> Unit,
    onOpenIpDialog: () -> Unit,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier
            .clip(RoundedCornerShape(16.dp))
            .background(Color(0xCC0D1117))
            .border(1.dp, Color(0x4400F0FF), RoundedCornerShape(16.dp))
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        // Trạng thái kết nối
        Row(verticalAlignment = Alignment.CenterVertically) {
            val statusColor = when (connectionStatus) {
                ConnectionStatus.CONNECTED -> Color(0xFF00FF88)
                ConnectionStatus.CONNECTING -> Color(0xFFFFD600)
                ConnectionStatus.ERROR -> Color(0xFFFF2A55)
                ConnectionStatus.DISCONNECTED -> Color.Gray
            }
            Box(
                modifier = Modifier
                    .size(10.dp)
                    .clip(CircleShape)
                    .background(statusColor)
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = connectionStatus.name,
                color = statusColor,
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = FontFamily.Monospace
            )
            if (connectionStatus == ConnectionStatus.CONNECTED && latencyMs > 0) {
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "${latencyMs}ms",
                    color = if (latencyMs < 120) Color(0xFF00FF88) else Color(0xFFFF9900),
                    fontSize = 12.sp,
                    fontFamily = FontFamily.Monospace
                )
            }
        }

        // Cấu hình IP và Âm thanh
        Row(verticalAlignment = Alignment.CenterVertically) {
            TextButton(onClick = onOpenIpDialog, contentPadding = PaddingValues(horizontal = 8.dp)) {
                Icon(Icons.Default.Settings, contentDescription = "Cấu hình IP", tint = Color(0xFF00F0FF), modifier = Modifier.size(16.dp))
                Spacer(modifier = Modifier.width(4.dp))
                Text(text = serverIp.substringBefore(":"), color = Color.White, fontSize = 11.sp, fontFamily = FontFamily.Monospace)
            }

            IconButton(onClick = onToggleTts, modifier = Modifier.size(32.dp)) {
                Icon(
                    imageVector = if (isTtsEnabled) Icons.Default.VolumeUp else Icons.Default.VolumeOff,
                    contentDescription = "Bật/Tắt Giọng Nói",
                    tint = if (isTtsEnabled) Color(0xFF00FF88) else Color.Gray,
                    modifier = Modifier.size(20.dp)
                )
            }
        }
    }
}

/**
 * Bảng điều khiển & Hiển thị chỉ dẫn phía dưới (HUD Bottom Panel)
 */
@Composable
private fun BottomControlPanel(
    guidanceText: String,
    priority: String,
    isStreaming: Boolean,
    isOcrLoading: Boolean,
    onToggleStream: () -> Unit,
    onReadOcr: () -> Unit,
    modifier: Modifier = Modifier
) {
    val isDanger = priority.equals("HIGH", ignoreCase = true) || priority.equals("DANGER", ignoreCase = true)
    val accentColor = if (isDanger) Color(0xFFFF2A55) else Color(0xFF00FF88)

    Column(modifier = modifier, horizontalAlignment = Alignment.CenterHorizontally) {
        // Banner hiển thị câu hướng dẫn bằng chữ to rõ
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(20.dp))
                .background(Color(0xEE090D16))
                .border(2.dp, accentColor, RoundedCornerShape(20.dp))
                .padding(16.dp)
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = if (isDanger) "⚠️" else "🧭",
                    fontSize = 28.sp,
                    modifier = Modifier.padding(end = 12.dp)
                )
                Text(
                    text = guidanceText,
                    color = Color.White,
                    fontSize = 17.sp,
                    fontWeight = FontWeight.SemiBold,
                    lineHeight = 24.sp
                )
            }
        }

        Spacer(modifier = Modifier.height(14.dp))

        // Nút điều khiển chính
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            // Nút Bắt đầu / Dừng quét
            Button(
                onClick = onToggleStream,
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (isStreaming) Color(0xFFE50914) else Color(0xFF00E676)
                ),
                shape = RoundedCornerShape(16.dp),
                modifier = Modifier.weight(1.5f).height(56.dp)
            ) {
                Icon(
                    imageVector = if (isStreaming) Icons.Default.Stop else Icons.Default.PlayArrow,
                    contentDescription = null,
                    tint = Color.Black
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = if (isStreaming) "DỪNG QUÉT" else "BẮT ĐẦU QUÉT",
                    color = Color.Black,
                    fontWeight = FontWeight.Bold,
                    fontSize = 15.sp
                )
            }

            // Nút đọc biển báo OCR
            Button(
                onClick = onReadOcr,
                enabled = !isOcrLoading,
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF00B0FF)),
                shape = RoundedCornerShape(16.dp),
                modifier = Modifier.weight(1f).height(56.dp)
            ) {
                if (isOcrLoading) {
                    CircularProgressIndicator(color = Color.White, modifier = Modifier.size(24.dp), strokeWidth = 2.5.dp)
                } else {
                    Icon(Icons.Default.MenuBook, contentDescription = null, tint = Color.Black)
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(text = "ĐỌC CHỮ", color = Color.Black, fontWeight = FontWeight.Bold, fontSize = 14.sp)
                }
            }
        }
    }
}

/**
 * Hộp thoại cấu hình IP Server AI
 */
@Composable
private fun IpConfigDialog(
    currentIp: String,
    onDismiss: () -> Unit,
    onSave: (String) -> Unit
) {
    var textIp by remember { mutableStateOf(currentIp) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Cấu hình IP Server AI", fontWeight = FontWeight.Bold) },
        text = {
            Column {
                Text("Nhập địa chỉ IP máy tính đang chạy Python backend (cùng mạng Wi-Fi):", fontSize = 13.sp)
                Spacer(modifier = Modifier.height(10.dp))
                OutlinedTextField(
                    value = textIp,
                    onValueChange = { textIp = it },
                    singleLine = true,
                    label = { Text("Địa chỉ IP:Cổng") }
                )
                Spacer(modifier = Modifier.height(10.dp))
                Text("Gợi ý cài đặt sẵn:", fontSize = 12.sp, color = Color.Gray)
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    AssistChip(onClick = { textIp = "127.0.0.1:8000" }, label = { Text("Cáp USB") })
                    AssistChip(onClick = { textIp = "192.168.1.49:8000" }, label = { Text("Wi-Fi") })
                    AssistChip(onClick = { textIp = "10.0.2.2:8000" }, label = { Text("Máy ảo") })
                }
            }
        },
        confirmButton = {
            Button(onClick = { onSave(textIp.trim()) }) {
                Text("Lưu kết nối")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Hủy")
            }
        }
    )
}

@Composable
private fun CameraPermissionDeniedContent(onRequestPermission: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize().background(Color(0xFF0F172A)).padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(Icons.Default.CameraAlt, contentDescription = null, tint = Color(0xFF00F0FF), modifier = Modifier.size(64.dp))
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            text = "Cần quyền Camera để Mũ Thông Minh hoạt động",
            color = Color.White,
            fontWeight = FontWeight.Bold,
            fontSize = 18.sp,
            textAlign = TextAlign.Center
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "Ứng dụng cần camera để quét và nhận diện vật cản phía trước theo thời gian thực.",
            color = Color.LightGray,
            fontSize = 14.sp,
            textAlign = TextAlign.Center
        )
        Spacer(modifier = Modifier.height(24.dp))
        Button(
            onClick = onRequestPermission,
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF00E676))
        ) {
            Text("CẤP QUYỀN CAMERA", color = Color.Black, fontWeight = FontWeight.Bold)
        }
    }
}
