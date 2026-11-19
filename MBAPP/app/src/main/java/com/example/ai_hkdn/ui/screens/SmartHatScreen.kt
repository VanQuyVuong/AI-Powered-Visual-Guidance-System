package com.example.ai_hkdn.ui.screens

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.MenuBook
import androidx.compose.material.icons.automirrored.filled.VolumeOff
import androidx.compose.material.icons.automirrored.filled.VolumeUp
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.example.ai_hkdn.data.model.DetectedObject
import com.example.ai_hkdn.edge.EdgeModelsManager
import com.example.ai_hkdn.edge.EdgeVisionPipeline
import com.example.ai_hkdn.service.*
import com.example.ai_hkdn.ui.camera.CameraPreviewView
import com.example.ai_hkdn.ui.hud.ARHudOverlay
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

enum class ProcessingMode {
    EDGE_NPU,      // Điện thoại tự chạy 4 AI models độc lập (Tắt laptop/Mất Wi-Fi vẫn chạy)
    SERVER_LAPTOP  // Gửi ảnh qua WebSocket lên Laptop xử lý
}

@Composable
fun SmartHatScreen() {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    val prefs = remember { context.getSharedPreferences("hkdn_edge_prefs", Context.MODE_PRIVATE) }

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted -> hasCameraPermission = isGranted }

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

    // Trạng thái cấu hình & vận hành
    var processingMode by remember {
        val saved = prefs.getString("processing_mode", ProcessingMode.EDGE_NPU.name)
        mutableStateOf(runCatching { ProcessingMode.valueOf(saved ?: "") }.getOrDefault(ProcessingMode.EDGE_NPU))
    }
    var serverIp by remember {
        mutableStateOf(prefs.getString("server_ip", "192.168.1.49:8000") ?: "192.168.1.49:8000")
    }
    var telemetryEnabled by remember {
        mutableStateOf(prefs.getBoolean("telemetry_enabled", true))
    }
    var isBlindHandsFreeMode by remember { mutableStateOf(false) }

    var isStreaming by remember { mutableStateOf(false) }
    var connectionStatus by remember { mutableStateOf(ConnectionStatus.DISCONNECTED) }
    var telemetryStatus by remember { mutableStateOf(TelemetryStatus.DISABLED) }
    var runtimeSummary by remember { mutableStateOf("Sẵn sàng NPU Độc lập") }
    var detectedObjects by remember { mutableStateOf<List<DetectedObject>>(emptyList()) }
    var guidanceText by remember { mutableStateOf("Sẵn sàng. Bấm BẮT ĐẦU QUÉT để chạy AI độc lập trên điện thoại.") }
    var guidancePriority by remember { mutableStateOf("LOW") }
    var latencyMs by remember { mutableLongStateOf(0L) }
    var isTtsEnabled by remember { mutableStateOf(true) }
    var isOcrLoading by remember { mutableStateOf(false) }
    var showConfigDialog by remember { mutableStateOf(false) }

    var latestJpegBytes by remember { mutableStateOf<ByteArray?>(null) }
    var latestEdgeSnapshotJpeg by remember { mutableStateOf<ByteArray?>(null) }
    var srcFrameWidth by remember { mutableIntStateOf(480) }
    var srcFrameHeight by remember { mutableIntStateOf(640) }

    val mainHandler = remember { android.os.Handler(android.os.Looper.getMainLooper()) }
    val edgeExecutor = remember { Executors.newSingleThreadExecutor() }
    val isEdgeBusy = remember { AtomicBoolean(false) }

    val ttsManager = remember { TtsManager(context) }
    val ocrService = remember { OcrService("http://$serverIp") }
    val modelsManager = remember { EdgeModelsManager(context) }
    val edgePipeline = remember { EdgeVisionPipeline(modelsManager) }

    val telemetryService = remember {
        TelemetryService(
            serverUrl = "ws://$serverIp/api/v1/vision/stream",
            onStatusChange = { status -> telemetryStatus = status }
        )
    }

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
                        if (isTtsEnabled) ttsManager.speak(text, priority)
                    }
                }
            },
            onStatusChange = { newStatus ->
                connectionStatus = newStatus
                if (processingMode == ProcessingMode.SERVER_LAPTOP) {
                    if (newStatus == ConnectionStatus.CONNECTED) {
                        guidanceText = "Đã kết nối Server Laptop! Đang quét..."
                        if (isTtsEnabled) ttsManager.speak("Đã kết nối máy chủ thành công", force = true)
                    } else if (newStatus == ConnectionStatus.DISCONNECTED && isStreaming) {
                        guidanceText = "⚠️ Mất kết nối máy chủ, hãy chuyển sang NPU Độc lập."
                        guidancePriority = "CRITICAL"
                        if (isTtsEnabled) ttsManager.speak("Mất kết nối máy chủ.", priority = "CRITICAL", force = true)
                    }
                }
            },
            onError = { errMsg ->
                if (processingMode == ProcessingMode.SERVER_LAPTOP && isStreaming) {
                    guidanceText = "⚠️ $errMsg"
                    guidancePriority = "HIGH"
                }
            }
        )
    }

    // Nạp sẵn 4 mô hình ONNX FP16 vào RAM/NPU ngay khi mở ứng dụng để bấm Quét là chạy tức thì
    LaunchedEffect(Unit) {
        coroutineScope.launch(Dispatchers.IO) {
            runCatching {
                val summary = edgePipeline.warmUp(preferNpu = true)
                mainHandler.post { runtimeSummary = summary }
            }.onFailure { err ->
                mainHandler.post { runtimeSummary = "Lỗi nạp NPU: ${err.message}" }
            }
        }
    }

    LaunchedEffect(serverIp) {
        val cleanHost = serverIp.trim()
            .removePrefix("http://").removePrefix("https://")
            .removePrefix("ws://").removePrefix("wss://")
        val isSecure = serverIp.startsWith("https://") || serverIp.startsWith("wss://") ||
            cleanHost.contains("ngrok") || cleanHost.contains("trycloudflare")
        val wsScheme = if (isSecure) "wss" else "ws"
        val httpScheme = if (isSecure) "https" else "http"

        webSocketService.updateServerUrl("$wsScheme://$cleanHost/api/v1/vision/stream")
        telemetryService.updateServerUrl("$wsScheme://$cleanHost/api/v1/vision/stream")
        ocrService.updateServerUrl("$httpScheme://$cleanHost")
    }

    DisposableEffect(Unit) {
        onDispose {
            webSocketService.disconnect()
            telemetryService.shutdown()
            ttsManager.shutdown()
            modelsManager.close()
            edgeExecutor.shutdown()
        }
    }

    fun toggleScanning() {
        if (isStreaming) {
            isStreaming = false
            isEdgeBusy.set(false)
            webSocketService.disconnect()
            telemetryService.setEnabled(false)
            ttsManager.stop()
            detectedObjects = emptyList()
            guidanceText = "Đã dừng quét."
            if (isTtsEnabled) ttsManager.speak("Đã dừng quét dẫn đường.", force = true)
        } else {
            if (processingMode == ProcessingMode.EDGE_NPU) {
                edgePipeline.resetState()
                telemetryService.setEnabled(telemetryEnabled)
                guidanceText = "Đang chạy độc lập trên điện thoại..."
                guidancePriority = "LOW"
                isStreaming = true
                if (isTtsEnabled) ttsManager.speak("Bắt đầu dẫn đường độc lập.", force = true)
            } else {
                telemetryService.setEnabled(false)
                webSocketService.connect()
                isStreaming = true
            }
        }
    }

    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        // 1. CameraX Preview + Frame Provider
        CameraPreviewView(
            isStreaming = isStreaming,
            useEdgeBitmap = (processingMode == ProcessingMode.EDGE_NPU),
            canProcessFrame = {
                if (processingMode == ProcessingMode.EDGE_NPU) !isEdgeBusy.get()
                else webSocketService.canSendFrame()
            },
            onFrameCaptured = { bytes, base64 ->
                latestJpegBytes = bytes
                if (isStreaming && processingMode == ProcessingMode.SERVER_LAPTOP) {
                    webSocketService.sendFrame(base64)
                }
            },
            onEdgeBitmapCaptured = { bitmap ->
                if (!isStreaming || processingMode != ProcessingMode.EDGE_NPU || !isEdgeBusy.compareAndSet(false, true)) {
                    bitmap.recycle()
                    return@CameraPreviewView
                }
                edgeExecutor.execute {
                    try {
                        // Lưu nhanh 1 ảnh mẫu JPEG cho nút ĐỌC CHỮ nếu người dùng bấm
                        val baos = ByteArrayOutputStream()
                        bitmap.compress(Bitmap.CompressFormat.JPEG, 70, baos)
                        latestEdgeSnapshotJpeg = baos.toByteArray()

                        val needVisuals = telemetryEnabled && telemetryService.isConnectedToLaptop()
                        val packet = edgePipeline.analyzeBitmap(
                            frameBitmap = bitmap,
                            preferNpu = true,
                            needTelemetryVisuals = needVisuals,
                        )
                        telemetryService.sendTelemetryIfReady(packet, bitmap.width, bitmap.height)

                        mainHandler.post {
                            if (!isStreaming) return@post
                            latencyMs = packet.processingMs.toLong()
                            runtimeSummary = packet.runtimeSummary
                            srcFrameWidth = bitmap.width
                            srcFrameHeight = bitmap.height
                            detectedObjects = packet.analysis.detectedObjects
                            val displayMsg = packet.analysis.displayText
                            val speechMsg = packet.analysis.guidance.text
                            val prio = packet.analysis.guidance.priority
                            if (displayMsg.isNotBlank()) {
                                guidanceText = displayMsg
                                guidancePriority = prio
                            }
                            if (speechMsg.isNotBlank() && isTtsEnabled) {
                                ttsManager.speak(speechMsg, prio)
                            }
                        }
                    } catch (e: Exception) {
                        mainHandler.post {
                            guidanceText = "Lỗi Edge AI: ${e.message}"
                        }
                    } finally {
                        if (!bitmap.isRecycled) bitmap.recycle()
                        isEdgeBusy.set(false)
                    }
                }
            },
            modifier = Modifier.fillMaxSize()
        )

        // 2. AR HUD Overlay
        ARHudOverlay(
            objects = detectedObjects,
            imageSourceWidth = srcFrameWidth,
            imageSourceHeight = srcFrameHeight,
            showRoi = isStreaming && !isBlindHandsFreeMode,
            showBoxes = isStreaming && !isBlindHandsFreeMode,
            modifier = Modifier.fillMaxSize()
        )

        if (isBlindHandsFreeMode) {
            // Chế độ Người Khiếm Thị / Túi Áo: Màn hình tối tiết kiệm pin, chạm bất kỳ đâu để Bật/Dừng
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color(0xEE04060A))
                    .clickable { toggleScanning() }
                    .padding(28.dp),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Icon(
                        imageVector = if (isStreaming) Icons.Default.Hearing else Icons.Default.TouchApp,
                        contentDescription = null,
                        tint = if (isStreaming) Color(0xFF00FF88) else Color(0xFF00F0FF),
                        modifier = Modifier.size(84.dp)
                    )
                    Spacer(modifier = Modifier.height(20.dp))
                    Text(
                        text = if (isStreaming) "ĐANG DẪN ĐƯỜNG ĐỘC LẬP" else "CHẠM MÀN HÌNH ĐỂ BẮT ĐẦU",
                        color = Color.White,
                        fontSize = 22.sp,
                        fontWeight = FontWeight.Bold,
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = guidanceText,
                        color = Color(0xFF00F0FF),
                        fontSize = 18.sp,
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(28.dp))
                    OutlinedButton(
                        onClick = { isBlindHandsFreeMode = false },
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = Color.LightGray)
                    ) {
                        Text("Thoát chế độ màn hình tối")
                    }
                }
            }
        } else {
            // 3. Top Status Bar
            TopStatusBar(
                processingMode = processingMode,
                connectionStatus = connectionStatus,
                telemetryStatus = telemetryStatus,
                telemetryEnabled = telemetryEnabled,
                runtimeSummary = runtimeSummary,
                latencyMs = latencyMs,
                isTtsEnabled = isTtsEnabled,
                onToggleTts = {
                    isTtsEnabled = !isTtsEnabled
                    if (!isTtsEnabled) ttsManager.stop()
                },
                onToggleHandsFree = { isBlindHandsFreeMode = true },
                onOpenConfigDialog = { showConfigDialog = true },
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .fillMaxWidth()
                    .statusBarsPadding()
                    .padding(12.dp)
            )

            // 4. Bottom Control Panel
            BottomControlPanel(
                guidanceText = guidanceText,
                priority = guidancePriority,
                isStreaming = isStreaming,
                isOcrLoading = isOcrLoading,
                onToggleStream = { toggleScanning() },
                onReadOcr = {
                    val bytes = latestEdgeSnapshotJpeg ?: latestJpegBytes
                    if (bytes != null) {
                        isOcrLoading = true
                        coroutineScope.launch {
                            ttsManager.speak("Đang đọc chữ...", force = true)
                            val result = ocrService.readTextFromImage(bytes)
                            isOcrLoading = false
                            result.onSuccess { text ->
                                guidanceText = "📖 $text"
                                ttsManager.speak(text, force = true)
                            }.onFailure {
                                guidanceText = "Đọc chữ cần kết nối máy chủ."
                                ttsManager.speak("Tính năng đọc chữ cần kết nối với máy tính.", force = true)
                            }
                        }
                    } else {
                        guidanceText = "Chưa có hình ảnh. Hãy bấm BẮT ĐẦU QUÉT trước."
                    }
                },
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .fillMaxWidth()
                    .navigationBarsPadding()
                    .padding(16.dp)
            )
        }

        if (showConfigDialog) {
            EdgeAndServerConfigDialog(
                currentMode = processingMode,
                currentIp = serverIp,
                currentTelemetry = telemetryEnabled,
                onDismiss = { showConfigDialog = false },
                onSave = { newMode, newIp, newTele ->
                    val modeChanged = (newMode != processingMode)
                    processingMode = newMode
                    serverIp = newIp
                    telemetryEnabled = newTele
                    prefs.edit()
                        .putString("processing_mode", newMode.name)
                        .putString("server_ip", newIp)
                        .putBoolean("telemetry_enabled", newTele)
                        .apply()

                    if (isStreaming) {
                        if (modeChanged) {
                            toggleScanning()
                        } else if (newMode == ProcessingMode.EDGE_NPU) {
                            telemetryService.setEnabled(newTele)
                        }
                    }
                    showConfigDialog = false
                }
            )
        }
    }
}

@Composable
private fun TopStatusBar(
    processingMode: ProcessingMode,
    connectionStatus: ConnectionStatus,
    telemetryStatus: TelemetryStatus,
    telemetryEnabled: Boolean,
    runtimeSummary: String,
    latencyMs: Long,
    isTtsEnabled: Boolean,
    onToggleTts: () -> Unit,
    onToggleHandsFree: () -> Unit,
    onOpenConfigDialog: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .clip(RoundedCornerShape(16.dp))
            .background(Color(0xCC0D1117))
            .border(1.dp, Color(0x4400F0FF), RoundedCornerShape(16.dp))
            .padding(horizontal = 12.dp, vertical = 8.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                val dotColor = if (processingMode == ProcessingMode.EDGE_NPU) {
                    Color(0xFF00FF88)
                } else when (connectionStatus) {
                    ConnectionStatus.CONNECTED -> Color(0xFF00FF88)
                    ConnectionStatus.CONNECTING -> Color(0xFFFFD600)
                    ConnectionStatus.ERROR -> Color(0xFFFF2A55)
                    ConnectionStatus.DISCONNECTED -> Color.Gray
                }
                Box(modifier = Modifier.size(10.dp).clip(CircleShape).background(dotColor))
                Spacer(modifier = Modifier.width(8.dp))
                Column {
                    Text(
                        text = if (processingMode == ProcessingMode.EDGE_NPU) "EDGE NPU ĐỘC LẬP" else "SERVER: ${connectionStatus.name}",
                        color = dotColor,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = FontFamily.Monospace
                    )
                    Text(
                        text = runtimeSummary,
                        color = Color(0xFFCBD5E1),
                        fontSize = 10.sp,
                        fontFamily = FontFamily.Monospace
                    )
                }
                if (latencyMs > 0) {
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "${latencyMs}ms",
                        color = if (latencyMs < 180) Color(0xFF00FF88) else Color(0xFFFF9900),
                        fontSize = 11.sp,
                        fontFamily = FontFamily.Monospace
                    )
                }
            }

            Row(verticalAlignment = Alignment.CenterVertically) {
                IconButton(onClick = onToggleHandsFree, modifier = Modifier.size(32.dp)) {
                    Icon(Icons.Default.VisibilityOff, contentDescription = "Chế độ Khiếm Thị", tint = Color(0xFF00F0FF), modifier = Modifier.size(19.dp))
                }
                IconButton(onClick = onOpenConfigDialog, modifier = Modifier.size(32.dp)) {
                    Icon(Icons.Default.Settings, contentDescription = "Cấu hình", tint = Color(0xFF00F0FF), modifier = Modifier.size(19.dp))
                }
                IconButton(onClick = onToggleTts, modifier = Modifier.size(32.dp)) {
                    Icon(
                        imageVector = if (isTtsEnabled) Icons.AutoMirrored.Filled.VolumeUp else Icons.AutoMirrored.Filled.VolumeOff,
                        contentDescription = "Âm thanh",
                        tint = if (isTtsEnabled) Color(0xFF00FF88) else Color.Gray,
                        modifier = Modifier.size(20.dp)
                    )
                }
            }
        }

        if (processingMode == ProcessingMode.EDGE_NPU && telemetryEnabled) {
            val teleText = when (telemetryStatus) {
                TelemetryStatus.LIVE_MIRROR -> "📡 Laptop Monitor: ĐANG ĐỒNG BỘ TRỰC TIẾP"
                TelemetryStatus.SEARCHING -> "📡 Laptop Monitor: Chờ Wi-Fi Laptop (vẫn chạy độc lập)"
                TelemetryStatus.DISABLED -> "📴 Chạy offline hoàn toàn"
            }
            Text(
                text = teleText,
                color = if (telemetryStatus == TelemetryStatus.LIVE_MIRROR) Color(0xFF00F0FF) else Color(0xFF94A3B8),
                fontSize = 10.sp,
                fontFamily = FontFamily.Monospace,
                modifier = Modifier.padding(top = 4.dp)
            )
        }
    }
}

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
    val isDanger = priority.equals("HIGH", ignoreCase = true) ||
        priority.equals("CRITICAL", ignoreCase = true) ||
        priority.equals("DANGER", ignoreCase = true)
    val accentColor = if (isDanger) Color(0xFFFF2A55) else Color(0xFF00FF88)

    Column(modifier = modifier, horizontalAlignment = Alignment.CenterHorizontally) {
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

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
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
                    Icon(Icons.AutoMirrored.Filled.MenuBook, contentDescription = null, tint = Color.Black)
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(text = "ĐỌC CHỮ", color = Color.Black, fontWeight = FontWeight.Bold, fontSize = 14.sp)
                }
            }
        }
    }
}

@Composable
private fun EdgeAndServerConfigDialog(
    currentMode: ProcessingMode,
    currentIp: String,
    currentTelemetry: Boolean,
    onDismiss: () -> Unit,
    onSave: (ProcessingMode, String, Boolean) -> Unit
) {
    var selectedMode by remember { mutableStateOf(currentMode) }
    var textIp by remember { mutableStateOf(currentIp) }
    var telemetryOn by remember { mutableStateOf(currentTelemetry) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Cấu hình NPU & Giám sát Laptop", fontWeight = FontWeight.Bold) },
        text = {
            Column {
                Text("1. Chế độ xử lý AI:", fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                Spacer(modifier = Modifier.height(6.dp))
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    FilterChip(
                        selected = selectedMode == ProcessingMode.EDGE_NPU,
                        onClick = { selectedMode = ProcessingMode.EDGE_NPU },
                        label = { Text("NPU Độc lập") },
                        modifier = Modifier.weight(1f)
                    )
                    FilterChip(
                        selected = selectedMode == ProcessingMode.SERVER_LAPTOP,
                        onClick = { selectedMode = ProcessingMode.SERVER_LAPTOP },
                        label = { Text("Server Laptop") },
                        modifier = Modifier.weight(1f)
                    )
                }

                Spacer(modifier = Modifier.height(10.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text("Gửi hình & kết quả xem trên Laptop", fontSize = 13.sp, modifier = Modifier.weight(1f))
                    Switch(checked = telemetryOn, onCheckedChange = { telemetryOn = it })
                }

                Spacer(modifier = Modifier.height(8.dp))
                OutlinedTextField(
                    value = textIp,
                    onValueChange = { textIp = it },
                    singleLine = true,
                    label = { Text("IP Laptop (khi cần theo dõi qua Wi-Fi)") }
                )
                Spacer(modifier = Modifier.height(6.dp))
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    AssistChip(onClick = { textIp = "127.0.0.1:8000" }, label = { Text("USB") })
                    AssistChip(onClick = { textIp = "192.168.1.49:8000" }, label = { Text("Wi-Fi") })
                }
            }
        },
        confirmButton = {
            Button(onClick = { onSave(selectedMode, textIp.trim(), telemetryOn) }) {
                Text("Lưu cài đặt")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("Đóng") }
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
            text = "Cần quyền Camera để Kính Thông Minh hoạt động",
            color = Color.White,
            fontWeight = FontWeight.Bold,
            fontSize = 18.sp,
            textAlign = TextAlign.Center
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "Ứng dụng cần camera để quét và cảnh báo vật cản trực tiếp trên điện thoại.",
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
