package com.example.ai_hkdn.service

import android.util.Log
import com.example.ai_hkdn.data.model.VisionResponse
import com.google.gson.Gson
import okhttp3.*
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

enum class ConnectionStatus {
    DISCONNECTED,
    CONNECTING,
    CONNECTED,
    ERROR
}

/**
 * Quản lý kết nối WebSocket thời gian thực tới Backend AI
 * Áp dụng Kỹ thuật Adaptive Backpressure để loại bỏ triệt để độ trễ (delay)
 */
class WebSocketService(
    private var serverUrl: String,
    private val onResult: (VisionResponse, Long) -> Unit,
    private val onStatusChange: (ConnectionStatus) -> Unit,
    private val onError: (String) -> Unit
) {
    private val tag = "WebSocketService"
    private val gson = Gson()
    private val client = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .writeTimeout(5, TimeUnit.SECONDS)
        .build()

    private var webSocket: WebSocket? = null
    var status: ConnectionStatus = ConnectionStatus.DISCONNECTED
        private set

    // Cờ kiểm soát Adaptive Backpressure:
    // CHỈ gửi frame mới khi server đã trả lời frame cũ -> Ngăn chặn 100% hiện tượng nghẽn hàng đợi (lag delay)
    private val isWaitingForResponse = AtomicBoolean(false)
    private var lastSendTime: Long = 0L

    fun updateServerUrl(newUrl: String) {
        if (serverUrl != newUrl) {
            disconnect()
            serverUrl = newUrl
        }
    }

    fun connect() {
        if (status == ConnectionStatus.CONNECTING || status == ConnectionStatus.CONNECTED) return

        setStatus(ConnectionStatus.CONNECTING)
        val request = try {
            Request.Builder().url(serverUrl).build()
        } catch (e: Exception) {
            setStatus(ConnectionStatus.ERROR)
            onError("URL Server không hợp lệ: ${e.message}")
            return
        }

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.i(tag, "✅ Kết nối WebSocket thành công tới: $serverUrl")
                setStatus(ConnectionStatus.CONNECTED)
                isWaitingForResponse.set(false)
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                val latency = System.currentTimeMillis() - lastSendTime
                isWaitingForResponse.set(false) // Giải phóng cờ, sẵn sàng cho frame tiếp theo

                try {
                    val visionResponse = gson.fromJson(text, VisionResponse::class.java)
                    if (visionResponse != null) {
                        onResult(visionResponse, latency)
                    }
                } catch (e: Exception) {
                    Log.e(tag, "Lỗi giải mã JSON phản hồi: ${e.message}")
                }
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                Log.w(tag, "WebSocket đang đóng: $reason (mã $code)")
                setStatus(ConnectionStatus.DISCONNECTED)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                Log.i(tag, "WebSocket đã ngắt kết nối.")
                setStatus(ConnectionStatus.DISCONNECTED)
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e(tag, "❌ Lỗi WebSocket: ${t.message}")
                setStatus(ConnectionStatus.ERROR)
                isWaitingForResponse.set(false)
                onError("Mất kết nối tới Server AI: ${t.localizedMessage ?: "Timeout"}")
            }
        })
    }

    /**
     * Gửi frame ảnh (Base64) lên server:
     * - Trả về `true` nếu frame được gửi đi.
     * - Trả về `false` nếu frame bị bỏ qua (đang đợi phản hồi từ frame trước) -> Giữ FPS mượt, không giật lag.
     */
    fun sendFrame(base64Image: String): Boolean {
        if (status != ConnectionStatus.CONNECTED) return false

        // Adaptive Backpressure: Nếu frame trước chưa xử lý xong, bỏ qua frame này (drop frame)
        if (isWaitingForResponse.get()) {
            // Nếu đã chờ quá 1500ms mà chưa có phản hồi -> Coi như timeout, reset cờ để không bị kẹt vĩnh viễn
            if (System.currentTimeMillis() - lastSendTime > 1500) {
                isWaitingForResponse.set(false)
            } else {
                return false // Drop frame một cách êm ái
            }
        }

        isWaitingForResponse.set(true)
        lastSendTime = System.currentTimeMillis()
        val success = webSocket?.send(base64Image) ?: false
        if (!success) {
            isWaitingForResponse.set(false)
        }
        return success
    }

    fun disconnect() {
        try {
            webSocket?.close(1000, "User stopped streaming")
        } catch (e: Exception) {
            // Ignore
        }
        webSocket = null
        isWaitingForResponse.set(false)
        setStatus(ConnectionStatus.DISCONNECTED)
    }

    private fun setStatus(newStatus: ConnectionStatus) {
        status = newStatus
        onStatusChange(newStatus)
    }
}
