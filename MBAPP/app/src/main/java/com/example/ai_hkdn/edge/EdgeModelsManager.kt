package com.example.ai_hkdn.edge

import android.content.Context
import android.util.Log
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import com.google.gson.Gson
import java.io.File
import java.io.FileOutputStream
import java.util.EnumSet

/**
 * Quản lý nạp và tăng tốc phần cứng cho 4 mô hình ONNX FP16 trực tiếp trên thiết bị (Redmi Note 14 Pro 5G):
 * - Giải nén 1 lần từ assets/models sang filesDir để mmap trực tiếp vào native C++ (tránh tràn JVM heap).
 * - Kích hoạt NNAPI FP16 (MediaTek Dimensity 7300 NPU/APU) kèm cơ chế tự lùi an toàn về XNNPACK CPU.
 */
class EdgeModelsManager(private val context: Context) {

    data class LoadedSession(
        val session: OrtSession,
        val providerLabel: String,
    )

    data class EdgeBundle(
        val manifest: ModelsManifest,
        val yoloe: LoadedSession,
        val obstacle: LoadedSession?,
        val depth: LoadedSession?,
        val segformer: LoadedSession?,
        val runtimeSummary: String,
    )

    private val tag = "EdgeModelsManager"
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val gson = Gson()
    private var bundle: EdgeBundle? = null

    fun getEnvironment(): OrtEnvironment = env

    @Synchronized
    fun ensureLoaded(preferNpu: Boolean = true): EdgeBundle {
        bundle?.let { return it }

        val manifestText = context.assets.open("models/models_manifest.json").bufferedReader().use { it.readText() }
        val manifest = gson.fromJson(manifestText, ModelsManifest::class.java)

        val modelsDir = File(context.filesDir, "edge_models").apply { mkdirs() }
        val yoloeFile = syncAssetModel(modelsDir, manifest.models["yoloe"]?.filename ?: "yoloe_26s_fp16.onnx", manifest.models["yoloe"]?.sizeBytes ?: 0L)
        val obstacleFile = syncAssetModel(modelsDir, manifest.models["obstacle"]?.filename ?: "obstacle_yolo11n_fp16.onnx", manifest.models["obstacle"]?.sizeBytes ?: 0L)
        val depthFile = syncAssetModel(modelsDir, manifest.models["depth"]?.filename ?: "depth_v2_fp16.onnx", manifest.models["depth"]?.sizeBytes ?: 0L)
        val segFile = syncAssetModel(modelsDir, manifest.models["segformer"]?.filename ?: "segformer_b0_fp16.onnx", manifest.models["segformer"]?.sizeBytes ?: 0L)

        val yoloeSess = openSessionWithFallback(yoloeFile, preferNpu = preferNpu)
        val obstacleSess = runCatching { openSessionWithFallback(obstacleFile, preferNpu = preferNpu) }.getOrNull()
        val depthSess = runCatching { openSessionWithFallback(depthFile, preferNpu = preferNpu) }.getOrNull()
        val segSess = runCatching { openSessionWithFallback(segFile, preferNpu = preferNpu) }.getOrNull()

        val activeCount = listOfNotNull(yoloeSess, obstacleSess, depthSess, segSess).size
        val usesNnapi = listOfNotNull(yoloeSess, obstacleSess, depthSess, segSess).any { it.providerLabel.contains("NNAPI") }
        val summary = if (usesNnapi) "NNAPI NPU + XNNPACK ($activeCount/4)" else "XNNPACK ARM64 ($activeCount/4)"

        return EdgeBundle(
            manifest = manifest,
            yoloe = yoloeSess,
            obstacle = obstacleSess,
            depth = depthSess,
            segformer = segSess,
            runtimeSummary = summary,
        ).also {
            bundle = it
            Log.i(tag, "Đã nạp xong $summary vào RAM thiết bị!")
        }
    }

    private fun syncAssetModel(dir: File, filename: String, expectedSize: Long): File {
        val outFile = File(dir, filename)
        if (outFile.exists() && (expectedSize <= 0L || outFile.length() == expectedSize)) {
            return outFile
        }
        context.assets.open("models/$filename").use { input ->
            FileOutputStream(outFile).use { output ->
                val buf = ByteArray(64 * 1024)
                while (true) {
                    val read = input.read(buf)
                    if (read <= 0) break
                    output.write(buf, 0, read)
                }
            }
        }
        return outFile
    }

    private fun openSessionWithFallback(modelFile: File, preferNpu: Boolean): LoadedSession {
        if (preferNpu) {
            try {
                val opts = OrtSession.SessionOptions().apply {
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                    setIntraOpNumThreads(4)
                    setInterOpNumThreads(2)
                    addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
                }
                val sess = env.createSession(modelFile.absolutePath, opts)
                return LoadedSession(sess, "NNAPI-FP16 (NPU)")
            } catch (t: Throwable) {
                Log.w(tag, "NNAPI không khả dụng cho ${modelFile.name}, tự động chuyển XNNPACK CPU: ${t.message}")
            }
        }

        val cpuOpts = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(4)
            setInterOpNumThreads(2)
        }
        return LoadedSession(env.createSession(modelFile.absolutePath, cpuOpts), "XNNPACK-CPU")
    }

    @Synchronized
    fun close() {
        bundle?.let { b ->
            runCatching { b.yoloe.session.close() }
            runCatching { b.obstacle?.session?.close() }
            runCatching { b.depth?.session?.close() }
            runCatching { b.segformer?.session?.close() }
        }
        bundle = null
    }
}
