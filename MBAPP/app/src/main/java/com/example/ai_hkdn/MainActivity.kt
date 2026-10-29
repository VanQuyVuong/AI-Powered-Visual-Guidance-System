package com.example.ai_hkdn

import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.example.ai_hkdn.ui.screens.SmartHatScreen
import com.example.ai_hkdn.ui.theme.AI_HKDNTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // Giữ màn hình luôn sáng khi đang quét dẫn đường
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setContent {
            AI_HKDNTheme {
                SmartHatScreen()
            }
        }
    }
}