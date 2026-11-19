package com.example.ai_hkdn.service

import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import com.example.ai_hkdn.edge.EdgeVisionPipeline
import com.google.gson.Gson
import okhttp3.*
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

enum class TelemetryStatus {
    DISABLED,
    SEARCHING,
    LIVE_MIRROR
}

/**
 * Kênh truyền Telemetry tùy chọn (Best-Effort) sang Laptop Web Monitor (/view-phone):
 * - Điện thoại tự xử lý toàn bộ 4 AI Models độc lập -> Laptop KHÔNG phải tính toán AI.
 * - Khi Laptop tắt hoặc ra ngoài vùng Wi-Fi, Telemetry tự ngắt êm ái và thử kết nối lại ngầm mỗi 6 giây;
 *   TUYỆT ĐỐI KHÔNG chặn suy luận hay báo mất kết nối làm gián đoạn người khiếm thị.
 */
class TelemetryService(
    private var serverUrl: String,
    private val onStatusChange: (TelemetryStatus) -> Unit = {},
) {
    private val tag = "TelemetryService"
    private val gson = Gson()
    private val mainHandler = Handler(Looper.getMainLooper())
    private val client = OkHttpClient.Builder()
        .connectTimeout(2, TimeUnit.SECONDS)
        .writeTimeout(3, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .build()

    private var webSocket: WebSocket? = null
    private val isConnecting = AtomicBoolean(false)
    private var isConnected = false
    private var isEnabled = false
    private var lastConnectAttemptMs = 0L
    private var lastFrameSentMs = 0L

    fun isConnectedToLaptop(): Boolean = isConnected

    fun updateServerUrl(newUrl: String) {
        if (serverUrl != newUrl) {
            serverUrl = newUrl
            if (isEnabled) {
                disconnectSocketOnly()
                tryConnectAsync()
            }
        }
    }

    fun setEnabled(enabled: Boolean) {
        isEnabled = enabled
        if (!enabled) {
            disconnectSocketOnly()
            postStatus(TelemetryStatus.DISABLED)
        } else {
            postStatus(TelemetryStatus.SEARCHING)
            tryConnectAsync()
        }
    }

    fun sendTelemetryIfReady(packet: EdgeVisionPipeline.EdgeFramePacket, imgW: Int, imgH: Int) {
        if (!isEnabled) return
        val now = System.currentTimeMillis()
        if (!isConnected) {
            if (!isConnecting.get() && (now - lastConnectAttemptMs) >= 6000L) {
                tryConnectAsync()
            }
            return
        }
        // Giới hạn ~8 FPS mirror lên Laptop để giữ pin & băng thông Wi-Fi nhẹ
        if ((now - lastFrameSentMs) < 120L) return
        lastFrameSentMs = now

        val a = packet.analysis
        val payload = hashMapOf<String, Any?>(
            "type" to "EDGE_TELEMETRY",
            "success" to true,
            "mode" to "EDGE_ON_DEVICE_NPU",
            "device" to "${Build.MODEL} (${packet.runtimeSummary})",
            "processing_ms" to packet.processingMs,
            "objects" to a.detectedObjects,
            "guidance" to mapOf(
                "text" to a.guidance.text,
                "display_text" to a.displayText,
                "priority" to a.guidance.priority,
            ),
            "navigation" to mapOf(
                "direction" to a.navigation.direction,
                "confidence" to a.navigation.confidence,
                "text" to a.navigation.text,
                "description" to a.displayText,
                "fusion_summary" to a.fusionSummaryMap,
            ),
            "img_size" to mapOf("width" to imgW, "height" to imgH),
            "frame_b64" to packet.frameB64,
            "depth_b64" to packet.depthB64,
        )

        val ok = runCatching { webSocket?.send(gson.toJson(payload)) == true }.getOrDefault(false)
        if (!ok) {
            isConnected = false
            postStatus(TelemetryStatus.SEARCHING)
        }
    }

    private fun tryConnectAsync() {
        if (!isEnabled || isConnecting.getAndSet(true)) return
        lastConnectAttemptMs = System.currentTimeMillis()
        val req = try {
            Request.Builder().url(serverUrl).build()
        } catch (e: Exception) {
            isConnecting.set(false)
            return
        }
        webSocket = client.newWebSocket(req, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                isConnecting.set(false)
                isConnected = true
                Log.i(tag, "Laptop Web Monitor đã sẵn sàng nhận Telemetry: $serverUrl")
                postStatus(TelemetryStatus.LIVE_MIRROR)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                isConnecting.set(false)
                isConnected = false
                if (isEnabled) postStatus(TelemetryStatus.SEARCHING)
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                isConnecting.set(false)
                isConnected = false
                if (isEnabled) postStatus(TelemetryStatus.SEARCHING)
            }
        })
    }

    fun shutdown() {
        isEnabled = false
        disconnectSocketOnly()
    }

    private fun disconnectSocketOnly() {
        runCatching { webSocket?.close(1000, "Edge telemetry paused") }
        webSocket = null
        isConnected = false
        isConnecting.set(false)
    }

    private fun postStatus(s: TelemetryStatus) {
        mainHandler.post { onStatusChange(s) }
    }
}
