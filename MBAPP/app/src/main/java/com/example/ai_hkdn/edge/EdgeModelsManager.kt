package com.example.ai_hkdn.edge

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import com.google.gson.Gson
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.EnumSet

/**
 * Quản lý nạp và tự động chọn bộ tăng tốc nhanh nhất cho 4 mô hình ONNX trên thiết bị thật
 * (MediaTek Dimensity 7300-Ultra NPU 655 + ARMv9 XNNPACK):
 * - Khắc phục xung đột thread pool giữa ORT và XNNPACK (`intraOpThreads = 1`, `allow_spinning = 0`, XNNPACK `4` luồng).
 * - Tự benchmark steady-state giữa NNAPI FP16 (NPU) và XNNPACK cho từng model và dọn file model cũ.
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
        val expectedFiles = manifest.models.values.map { it.filename }.toSet()
        modelsDir.listFiles()?.forEach { f ->
            val baseName = f.name.removeSuffix(".sha")
            if (baseName !in expectedFiles) {
                runCatching { f.delete() }
            }
        }

        val yoloeMeta = manifest.models["yoloe"]
        val obsMeta = manifest.models["obstacle"]
        val depthMeta = manifest.models["depth"]
        val segMeta = manifest.models["segformer"]

        val yoloeFile = syncAssetModel(modelsDir, yoloeMeta?.filename ?: "yoloe_26s.onnx", yoloeMeta?.sizeBytes ?: 0L, yoloeMeta?.sha256 ?: "")
        val obsFile = syncAssetModel(modelsDir, obsMeta?.filename ?: "obstacle_yolo11n.onnx", obsMeta?.sizeBytes ?: 0L, obsMeta?.sha256 ?: "")
        val depthFile = syncAssetModel(modelsDir, depthMeta?.filename ?: "depth_v2_fp16.onnx", depthMeta?.sizeBytes ?: 0L, depthMeta?.sha256 ?: "")
        val segFile = syncAssetModel(modelsDir, segMeta?.filename ?: "segformer_b0.onnx", segMeta?.sizeBytes ?: 0L, segMeta?.sha256 ?: "")

        val ySz = manifest.yoloImgsz.toLong()
        val dSz = manifest.depthImgsz.toLong()
        val sSz = manifest.segImgsz.toLong()

        val yoloeSess = openFastestSession(yoloeFile, longArrayOf(1L, 3L, ySz, ySz), preferNnapi = preferNpu)
        val obstacleSess = runCatching { openFastestSession(obsFile, longArrayOf(1L, 3L, ySz, ySz), preferNnapi = preferNpu) }
            .onFailure { Log.e(tag, "Lỗi nạp ObstacleYOLO11n: ${it.message}") }
            .getOrNull()
        val depthSess = runCatching { openFastestSession(depthFile, longArrayOf(1L, 3L, dSz, dSz), preferNnapi = false) }
            .onFailure { Log.e(tag, "Lỗi nạp DepthV2: ${it.message}") }
            .getOrNull()
        val segSess = runCatching { openFastestSession(segFile, longArrayOf(1L, 3L, sSz, sSz), preferNnapi = preferNpu) }
            .onFailure { Log.e(tag, "Lỗi nạp SegFormer: ${it.message}") }
            .getOrNull()

        val loaded = listOfNotNull(yoloeSess, obstacleSess, depthSess, segSess)
        val activeCount = loaded.size
        val npuCount = loaded.count { it.providerLabel.contains("NNAPI") }
        val summary = if (npuCount > 0) "NNAPI NPU ($npuCount) + XNNPACK ($activeCount/4)" else "XNNPACK ARM64 ($activeCount/4)"

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

    private fun syncAssetModel(dir: File, filename: String, expectedSize: Long, expectedSha: String): File {
        val outFile = File(dir, filename)
        val shaFile = File(dir, "$filename.sha")
        val savedSha = if (shaFile.exists()) shaFile.readText().trim() else ""
        if (outFile.exists() &&
            (expectedSize <= 0L || outFile.length() == expectedSize) &&
            (expectedSha.isEmpty() || savedSha == expectedSha)
        ) {
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
        if (expectedSha.isNotEmpty()) {
            shaFile.writeText(expectedSha)
        }
        return outFile
    }

    private fun openFastestSession(
        modelFile: File,
        inputShape: LongArray,
        preferNnapi: Boolean,
    ): LoadedSession {
        val xnnSess = createXnnpackSession(modelFile)
        val xnnMs = benchmarkSessionMin(xnnSess, inputShape)

        if (!preferNnapi) {
            Log.i(tag, "${modelFile.name} -> XNNPACK=${xnnMs}ms (chọn XNNPACK)")
            return LoadedSession(xnnSess, "XNNPACK (${xnnMs}ms)")
        }

        val nnapiSess = try {
            val opts = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setIntraOpNumThreads(1)
                setInterOpNumThreads(1)
                addConfigEntry("session.intra_op.allow_spinning", "0")
                addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
            }
            env.createSession(modelFile.absolutePath, opts)
        } catch (t: Throwable) {
            Log.w(tag, "NNAPI không khả dụng cho ${modelFile.name}: ${t.message}")
            null
        }

        if (nnapiSess == null) {
            Log.i(tag, "${modelFile.name} -> XNNPACK=${xnnMs}ms")
            return LoadedSession(xnnSess, "XNNPACK (${xnnMs}ms)")
        }

        val nnapiMs = try {
            benchmarkSessionMin(nnapiSess, inputShape)
        } catch (t: Throwable) {
            Log.w(tag, "NNAPI lỗi chạy thử ${modelFile.name}: ${t.message}")
            Long.MAX_VALUE
        }

        Log.i(tag, "Benchmark ${modelFile.name}: NNAPI=${nnapiMs}ms vs XNNPACK=${xnnMs}ms")
        return if (nnapiMs <= (xnnMs * 1.15f).toLong() && nnapiMs < Long.MAX_VALUE) {
            runCatching { xnnSess.close() }
            LoadedSession(nnapiSess, "NNAPI-NPU (${nnapiMs}ms)")
        } else {
            runCatching { nnapiSess.close() }
            LoadedSession(xnnSess, "XNNPACK (${xnnMs}ms)")
        }
    }

    private fun createXnnpackSession(modelFile: File): OrtSession {
        // Lưu ý từ cảnh báo ORT: khi dùng XNNPACK 4 luồng phải đặt intra_op_num_threads = 1 và tắt spinning để không tranh chấp thread pool!
        return try {
            val opts = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setIntraOpNumThreads(1)
                setInterOpNumThreads(1)
                addConfigEntry("session.intra_op.allow_spinning", "0")
                addXnnpack(mapOf("intra_op_num_threads" to "4"))
            }
            env.createSession(modelFile.absolutePath, opts)
        } catch (t: Throwable) {
            Log.w(tag, "XNNPACK ALL_OPT fallback -> BASIC_OPT+XNNPACK cho ${modelFile.name}: ${t.message}")
            try {
                val basicXnn = OrtSession.SessionOptions().apply {
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
                    setIntraOpNumThreads(1)
                    setInterOpNumThreads(1)
                    addConfigEntry("session.intra_op.allow_spinning", "0")
                    addXnnpack(mapOf("intra_op_num_threads" to "4"))
                }
                env.createSession(modelFile.absolutePath, basicXnn)
            } catch (t2: Throwable) {
                val safeOpts = OrtSession.SessionOptions().apply {
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
                    setIntraOpNumThreads(4)
                }
                env.createSession(modelFile.absolutePath, safeOpts)
            }
        }
    }

    private fun benchmarkSessionMin(sess: OrtSession, shape: LongArray): Long {
        val elems = shape.fold(1L) { acc, v -> acc * v }.toInt()
        val buf = ByteBuffer.allocateDirect(elems * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        val inName = sess.inputNames.first()
        var bestMs = Long.MAX_VALUE
        // 2 lần làm nóng + 3 lần đo lấy tốc độ thực tế ổn định nhất
        for (i in 0 until 5) {
            buf.rewind()
            val t0 = System.nanoTime()
            OnnxTensor.createTensor(env, buf, shape).use { t ->
                sess.run(mapOf(inName to t)).use { }
            }
            val ms = (System.nanoTime() - t0) / 1_000_000L
            if (i >= 2 && ms < bestMs) {
                bestMs = ms
            }
        }
        return bestMs
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
