package com.example.ai_hkdn.service

import android.content.Context
import android.speech.tts.TextToSpeech
import android.util.Log
import java.util.Locale

/**
 * Trình quản lý giọng nói tiếng Việt Native (TTS)
 * - Tối ưu chống nói đè (anti-overlap)
 * - Ưu tiên cảnh báo nguy cấp (HIGH/DANGER)
 */
class TtsManager(context: Context) : TextToSpeech.OnInitListener {

    private val tag = "TtsManager"
    private var tts: TextToSpeech? = TextToSpeech(context.applicationContext, this)
    private var isInitialized = false

    private var lastSpokenText: String = ""
    private var lastSpokenTime: Long = 0L

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts?.setLanguage(Locale("vi", "VN"))
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.w(tag, "Tiếng Việt không hỗ trợ đầy đủ trên máy này, dùng ngôn ngữ mặc định")
                tts?.setLanguage(Locale.getDefault())
            }
            tts?.setSpeechRate(1.05f) // Tốc độ hơi nhanh một chút để phản xạ kịp tình huống
            tts?.setPitch(1.0f)
            isInitialized = true
            Log.i(tag, "Khởi tạo TTS tiếng Việt thành công!")
        } else {
            Log.e(tag, "Khởi tạo TTS thất bại với mã: $status")
        }
    }

    /**
     * Đọc câu cảnh báo/chỉ đường thông minh:
     * - Nếu là cảnh báo khẩn (force / HIGH): Ngắt câu cũ đọc ngay lập tức!
     * - Nếu là cảnh báo thường: Kiểm tra nếu câu giống hệt vừa đọc trong 2.5s thì bỏ qua để không lặp lại gây khó chịu.
     */
    fun speak(text: String, priority: String = "LOW", force: Boolean = false) {
        if (!isInitialized || text.isBlank()) return

        val now = System.currentTimeMillis()
        val isSafeMessage = text.contains("an toàn", ignoreCase = true)
        val wasSafeMessage = lastSpokenText.contains("an toàn", ignoreCase = true)
        val isDanger = priority.equals("DANGER", ignoreCase = true) || priority.equals("HIGH", ignoreCase = true) || !isSafeMessage

        // 1. Nếu là thông báo "Phía trước an toàn": chỉ nhắc lại tối đa mỗi 8 giây và không bao giờ ngắt lời cảnh báo vật cản
        if (!force && isSafeMessage) {
            if (tts?.isSpeaking == true) return
            if (wasSafeMessage && (now - lastSpokenTime) < 8000) return
            if (!wasSafeMessage && (now - lastSpokenTime) < 3500) return
        }

        // 2. Chống lặp lại cùng một câu cảnh báo vật cản trong 2.5 giây
        if (!force && text == lastSpokenText && (now - lastSpokenTime) < 2500) {
            return
        }

        // 3. Nếu đang đọc câu "an toàn" hoặc lời chào mà phát hiện vật cản (ghế, chai, bàn, tường...) -> NGẮT NGAY để đọc vật cản!
        val shouldFlushImmediately = force || (isDanger && (wasSafeMessage || text != lastSpokenText))

        if (!shouldFlushImmediately && tts?.isSpeaking == true && (now - lastSpokenTime) < 1800) {
            return
        }

        lastSpokenText = text
        lastSpokenTime = now

        val queueMode = if (shouldFlushImmediately) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD
        tts?.speak(text, queueMode, null, "UTTERANCE_$now")
    }

    fun stop() {
        tts?.stop()
    }

    fun shutdown() {
        tts?.stop()
        tts?.shutdown()
        tts = null
    }
}
