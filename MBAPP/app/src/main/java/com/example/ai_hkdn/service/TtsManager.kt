package com.example.ai_hkdn.service

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.tts.TextToSpeech
import android.util.Log
import java.util.Locale

/**
 * Trình quản lý giọng nói tiếng Việt Native (TTS) kết hợp Phản hồi Xúc giác (Haptic Vibration)
 * - Tối ưu chống nói đè (anti-overlap)
 * - Ưu tiên cảnh báo nguy cấp (HIGH/CRITICAL)
 * - Rung theo mã nhịp giúp người khiếm thị nhận biết nguy hiểm ngay cả ở nơi ồn ào
 */
class TtsManager(context: Context) : TextToSpeech.OnInitListener {

    private val tag = "TtsManager"
    private val appContext = context.applicationContext
    private var tts: TextToSpeech? = TextToSpeech(appContext, this)
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
     * Phát phản hồi xúc giác (rung) tương ứng với mức độ nguy hiểm:
     * - CRITICAL / Quay đầu: Rung 3 nhịp dồn dập (SOS)
     * - Cầu thang / Bậc thềm / Ổ gà: Rung 2 nhịp mạnh
     * - HIGH / Rẽ trái / Rẽ phải: Rung 1 nhịp dứt khoát
     */
    private fun triggerHapticAlert(text: String, priority: String) {
        try {
            val lower = text.lowercase()
            if (lower.contains("an toàn")) return

            val vibrator: Vibrator? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                val vm = appContext.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager
                vm?.defaultVibrator
            } else {
                @Suppress("DEPRECATION")
                appContext.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
            }
            if (vibrator == null || !vibrator.hasVibrator()) return

            val pattern: LongArray = when {
                priority.equals("CRITICAL", ignoreCase = true) || lower.contains("quay đầu") ->
                    longArrayOf(0, 180, 80, 180, 80, 260)
                lower.contains("cầu thang") || lower.contains("bậc") || lower.contains("ổ gà") ->
                    longArrayOf(0, 220, 100, 220)
                else ->
                    longArrayOf(0, 140)
            }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vibrator.vibrate(VibrationEffect.createWaveform(pattern, -1))
            } else {
                @Suppress("DEPRECATION")
                vibrator.vibrate(pattern, -1)
            }
        } catch (e: Exception) {
            Log.w(tag, "Không thể kích hoạt rung: ${e.message}")
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
        val isSafeMessage = text.contains("an toàn", ignoreCase = true) ||
            text.contains("thông thoáng", ignoreCase = true)
        val wasSafeMessage = lastSpokenText.contains("an toàn", ignoreCase = true) ||
            lastSpokenText.contains("thông thoáng", ignoreCase = true)
        val isTransitionToSafe = isSafeMessage && !wasSafeMessage && lastSpokenText.isNotBlank()
        val isDanger = priority.equals("CRITICAL", ignoreCase = true) ||
            priority.equals("DANGER", ignoreCase = true) ||
            priority.equals("HIGH", ignoreCase = true) || !isSafeMessage

        // 1. Xử lý thông báo "Phía trước an toàn, đi thẳng":
        // - Nếu VỪA RẼ TRÁI/PHẢI XONG (chuyển từ cảnh báo vật cản sang an toàn - isTransitionToSafe):
        //   KHÔNG BAO GIỜ bỏ qua! Cho phép xếp hàng (QUEUE_ADD) ngay sau câu nhắc rẽ để đọc nối tiếp "Phía trước an toàn, đi thẳng."!
        // - Nếu đang đi thẳng bình thường (wasSafeMessage == true): chỉ nhắc lại tối đa mỗi 6 giây.
        if (!force && isSafeMessage) {
            if (isTransitionToSafe) {
                if ((now - lastSpokenTime) < 500) return
            } else {
                if (tts?.isSpeaking == true) return
                if (wasSafeMessage && (now - lastSpokenTime) < 6000) return
            }
        }

        // 2. Chống lặp lại cùng một câu cảnh báo vật cản trong 2.5 giây
        if (!force && text == lastSpokenText && (now - lastSpokenTime) < 2500) {
            return
        }

        // 3. Nếu đang đọc câu "an toàn" hoặc lời chào mà phát hiện vật cản (ghế, chai, bàn, tường...) -> NGẮT NGAY để đọc vật cản!
        val shouldFlushImmediately = force || (isDanger && (wasSafeMessage || text != lastSpokenText))

        if (!shouldFlushImmediately && !isTransitionToSafe && tts?.isSpeaking == true && (now - lastSpokenTime) < 1800) {
            return
        }

        lastSpokenText = text
        lastSpokenTime = now

        if (isDanger) {
            triggerHapticAlert(text, priority)
        }

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
