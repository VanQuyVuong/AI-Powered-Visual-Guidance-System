package com.example.ai_hkdn.service

import android.util.Log
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.concurrent.TimeUnit

data class OcrResponse(
    @SerializedName("success") val success: Boolean,
    @SerializedName("text") val text: String? = null,
    @SerializedName("detail") val detail: String? = null
)

class OcrService(private var serverBaseUrl: String) {

    private val tag = "OcrService"
    private val gson = Gson()
    private val client = OkHttpClient.Builder()
        .connectTimeout(8, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .build()

    fun updateServerUrl(newBaseUrl: String) {
        serverBaseUrl = newBaseUrl
    }

    suspend fun readTextFromImage(jpegBytes: ByteArray): Result<String> = withContext(Dispatchers.IO) {
        try {
            val url = "$serverBaseUrl/api/v1/vision/read_text"
            val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart(
                    "image",
                    "frame_ocr.jpg",
                    jpegBytes.toRequestBody("image/jpeg".toMediaTypeOrNull())
                )
                .build()

            val request = Request.Builder()
                .url(url)
                .post(requestBody)
                .build()

            val response = client.newCall(request).execute()
            val bodyString = response.body?.string() ?: ""

            if (!response.isSuccessful) {
                return@withContext Result.failure(Exception("Lỗi máy chủ HTTP ${response.code}: $bodyString"))
            }

            val ocrRes = gson.fromJson(bodyString, OcrResponse::class.java)
            if (ocrRes != null && ocrRes.success && !ocrRes.text.isNullOrBlank()) {
                Result.success(ocrRes.text)
            } else {
                Result.success("Không tìm thấy chữ hoặc biển báo nào rõ ràng.")
            }
        } catch (e: Exception) {
            Log.e(tag, "Lỗi gọi OCR API: ${e.message}")
            Result.failure(e)
        }
    }
}
