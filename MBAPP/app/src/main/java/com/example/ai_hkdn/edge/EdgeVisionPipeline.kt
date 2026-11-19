package com.example.ai_hkdn.edge

import ai.onnxruntime.OnnxTensor
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.util.Base64
import android.util.Log
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

/**
 * Pipeline thị giác AI chạy hoàn toàn độc lập trên thiết bị (Redmi Note 14 Pro 5G NPU/CPU):
 * 1. YOLOE-26s-FP16 (37 lớp mở) - mỗi frame
 * 2. ObstacleYOLO11n-FP16 (8 lớp ổ gà/hố ga/vật cản đường) - mỗi frame
 * 3. Depth-Anything-V2-Small-FP16 (Metric Z mét) - tự thích nghi chu kỳ theo tốc độ thiết bị
 * 4. SegFormer-B0-FP16 (ADE20K 150 lớp bề mặt) - mỗi 3 frame
 */
class EdgeVisionPipeline(
    private val modelsManager: EdgeModelsManager,
    private val decisionEngine: EdgeDecisionEngine = EdgeDecisionEngine(),
) {
    data class EdgeFramePacket(
        val analysis: EdgeAnalysisOutput,
        val processingMs: Float,
        val runtimeSummary: String,
        val frameB64: String?,
        val depthB64: String?,
    )

    private data class RawBoxDet(
        val classId: Int,
        val rawName: String,
        val normName: String,
        val confidence: Float,
        val x1: Int,
        val y1: Int,
        val x2: Int,
        val y2: Int,
    )

    private val tag = "EdgeVisionPipeline"

    // Preallocated direct FloatBuffers (Zero-GC per frame)
    private val yoloBuf: FloatBuffer = ByteBuffer.allocateDirect(1 * 3 * 320 * 320 * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val depthBuf: FloatBuffer = ByteBuffer.allocateDirect(1 * 3 * 252 * 252 * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val segBuf: FloatBuffer = ByteBuffer.allocateDirect(1 * 3 * 256 * 256 * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer()

    private val yoloPixels = IntArray(320 * 320)
    private val depthPixels = IntArray(252 * 252)
    private val segPixels = IntArray(256 * 256)

    // Bộ đệm tái sử dụng bitmap letterbox 320x320
    private val letterboxBmp: Bitmap = Bitmap.createBitmap(320, 320, Bitmap.Config.ARGB_8888)
    private val letterboxCanvas = Canvas(letterboxBmp)

    // Cache kết quả Depth & SegFormer mới nhất cho lập lịch xen kẽ
    private var latestDepthMap: FloatArray? = null
    private var depthW = 252
    private var depthH = 252
    private var lastDepthLatencyMs = 0L

    private var latestSegIds64: IntArray? = null
    private val segSw = 64
    private val segSh = 64
    private var latestSegHazardsNorm: List<SegNormHazard> = emptyList()
    private var cycleCount = 0

    private data class SegNormHazard(
        val className: String,
        val classNameVn: String,
        val x1n: Float,
        val y1n: Float,
        val x2n: Float,
        val y2n: Float,
        val distM: Float,
    )

    fun warmUp(preferNpu: Boolean = true): String {
        val bundle = modelsManager.ensureLoaded(preferNpu)
        return bundle.runtimeSummary
    }

    fun resetState() {
        cycleCount = 0
        latestDepthMap = null
        latestSegIds64 = null
        latestSegHazardsNorm = emptyList()
        decisionEngine.reset()
    }

    fun analyzeBitmap(
        frameBitmap: Bitmap,
        preferNpu: Boolean = true,
        needTelemetryVisuals: Boolean = false,
    ): EdgeFramePacket {
        val t0 = System.nanoTime()
        val bundle = modelsManager.ensureLoaded(preferNpu)
        val manifest = bundle.manifest
        val env = modelsManager.getEnvironment()
        val imgW = frameBitmap.width
        val imgH = frameBitmap.height

        cycleCount++
        val runDepth = (latestDepthMap == null) || (lastDepthLatencyMs <= 110L) || (cycleCount % 2 == 1)
        val runSeg = (latestSegIds64 == null) || (cycleCount % 3 == 0)

        // 1. SegFormer (mỗi 3 chu kỳ, dùng trước cho Ground/Floor/Stairs validation)
        if (runSeg && bundle.segformer != null) {
            runSegFormer(frameBitmap, bundle, manifest, env)
        }

        // 2. Depth Anything V2 (252x252 Metric Meters)
        if (runDepth && bundle.depth != null) {
            val td0 = System.nanoTime()
            runDepthEstimator(frameBitmap, bundle, manifest, env)
            lastDepthLatencyMs = (System.nanoTime() - td0) / 1_000_000L
        }

        val dmap = latestDepthMap
        val threeLanes = DistanceEstimator.analyzeThreeLanesDepth(dmap, depthW, depthH)

        // 3. YOLOE-26s-Seg (37 Open-Vocabulary guidance classes)
        val yoloSz = manifest.yoloImgsz.coerceAtLeast(320)
        val (scale, padX, padY) = prepareYoloLetterboxTensor(frameBitmap, yoloSz)
        val yoloeRaw = runYoloSession(
            session = bundle.yoloe,
            env = env,
            names = manifest.yoloeNames,
            imgsz = yoloSz,
            confThresh = 0.28f,
            srcW = imgW,
            srcH = imgH,
            scale = scale,
            padX = padX,
            padY = padY,
        )

        // 4. ObstacleYOLO11n (8 lớp hố ga/ổ gà/hố sâu) - dùng chung yoloBuf đã chuẩn bị!
        val obsRaw = if (bundle.obstacle != null) {
            runYoloSession(
                session = bundle.obstacle,
                env = env,
                names = manifest.obstacleNames,
                imgsz = yoloSz,
                confThresh = 0.34f,
                srcW = imgW,
                srcH = imgH,
                scale = scale,
                padX = padX,
                padY = padY,
            )
        } else {
            emptyList()
        }

        val candidates = ArrayList<EdgeCandidate>(yoloeRaw.size + obsRaw.size + 4)

        // Chuyển YOLOE detections -> EdgeCandidate kèm lọc gương & mặt đường
        for (det in yoloeRaw) {
            val cname = det.normName
            val box = EdgeBox(det.x1, det.y1, det.x2, det.y2)
            val areaRatio = (box.width.toFloat() * box.height.toFloat()) / max(1f, imgW.toFloat() * imgH.toFloat())
            val bhRatio = box.height.toFloat() / max(1f, imgH.toFloat())

            if ((cname == "mirror" || cname == "glass_door") && areaRatio < 0.08f && bhRatio < 0.35f) continue
            if (cname == "stairs" && areaRatio < 0.08f) continue
            if ((cname == "bed" || cname == "couch") && areaRatio < 0.12f) continue
            if (cname == "pothole" && !isValidPotholeRegion(box, imgW, imgH, manifest.segformerId2Label)) continue

            val pos = DistanceEstimator.determinePosition(box.x1, box.x2, imgW)
            var inCorridor = DistanceEstimator.isInRoi(box, imgW, imgH)
            val distM = when {
                cname == "mirror" || cname == "glass_door" -> {
                    if (pos == "center" && (cname == "glass_door" || bhRatio >= 0.45f)) inCorridor = true
                    DistanceEstimator.estimateSurfaceRingDepth(box, dmap, depthW, depthH, imgW, imgH)
                }
                cname == "pothole" -> {
                    inCorridor = true
                    DistanceEstimator.estimatePotholeDistance(box, dmap, depthW, depthH, imgW, imgH)
                }
                else -> {
                    val pinhole = DistanceEstimator.estimatePinholeDistanceM(cname, box, imgW, imgH)
                    val metric = DistanceEstimator.getBoxMedianDepth(box, dmap, depthW, depthH, imgW, imgH)
                    DistanceEstimator.resolveCalibratedDistance(cname, metric, pinhole, box, imgW, imgH)
                }
            }
            val overlapsCenter = (box.x1 < 0.58f * imgW && box.x2 > 0.42f * imgW)
            val inPath = (pos == "center") || overlapsCenter || (inCorridor && pos == "center")

            candidates.add(
                EdgeCandidate(
                    className = cname,
                    classNameVn = DistanceEstimator.getVietnameseName(cname),
                    confidence = max(det.confidence, 0.42f),
                    position = pos,
                    distanceM = distM,
                    inRoi = inCorridor || inPath,
                    isDanger = false,
                    boundingBox = box,
                    sourceModel = "YOLOE-26s-Edge",
                )
            )
        }

        // Chuyển ObstacleYOLO11n detections -> EdgeCandidate
        var auxUsedCount = 0
        for (det in obsRaw) {
            val cname = det.normName
            val box = EdgeBox(det.x1, det.y1, det.x2, det.y2)
            val isGroundDefect = cname in DistanceEstimator.GROUND_DEFECT_CLASSES
            if (isGroundDefect && !isValidPotholeRegion(box, imgW, imgH, manifest.segformerId2Label)) continue
            if (!isGroundDefect && det.confidence < 0.42f) continue

            val distM = if (isGroundDefect) {
                DistanceEstimator.estimatePotholeDistance(box, dmap, depthW, depthH, imgW, imgH)
            } else {
                DistanceEstimator.getBoxMedianDepth(box, dmap, depthW, depthH, imgW, imgH) ?: 2.5f
            }
            if ((isGroundDefect && distM > 3.2f) || (!isGroundDefect && distM > 2.4f)) continue

            auxUsedCount++
            val pos = DistanceEstimator.determinePosition(box.x1, box.x2, imgW)
            candidates.add(
                EdgeCandidate(
                    className = cname,
                    classNameVn = DistanceEstimator.getVietnameseName(cname),
                    confidence = max(det.confidence, 0.42f),
                    position = pos,
                    distanceM = distM,
                    inRoi = true,
                    isDanger = false,
                    boundingBox = box,
                    sourceModel = "ObstacleYOLO11n-Edge",
                )
            )
        }

        // Bổ sung cấu trúc SegFormer (cầu thang, cửa, cột, gương) nếu chưa có box gần che phủ
        val knownNearBoxes = candidates.filter { it.distanceM <= 2.25f }.map { it.boundingBox }
        var segAddedCount = 0
        for (sh in latestSegHazardsNorm) {
            val box = EdgeBox(
                (sh.x1n * imgW).toInt(),
                (sh.y1n * imgH).toInt(),
                (sh.x2n * imgW).toInt().coerceAtLeast(1),
                (sh.y2n * imgH).toInt().coerceAtLeast(1),
            )
            if (isCoveredByBoxes(box, knownNearBoxes, 0.35f)) continue
            val pos = DistanceEstimator.determinePosition(box.x1, box.x2, imgW)
            if (sh.className == "wall" && sh.distM > 1.35f) continue
            segAddedCount++
            candidates.add(
                EdgeCandidate(
                    className = sh.className,
                    classNameVn = sh.classNameVn,
                    confidence = 0.80f,
                    position = pos,
                    distanceM = sh.distM,
                    inRoi = true,
                    isDanger = false,
                    boundingBox = box,
                    sourceModel = "SegFormer-B0-Edge",
                )
            )
        }

        // Safety Inversion P0 từ Depth Map khi ở giữa có vật thể sát <= 0.98m chưa được YOLO bắt
        val hasCenterKnown = candidates.any { it.boundingBox.x1 < 0.65f * imgW && it.boundingBox.x2 > 0.35f * imgW && it.distanceM <= 2.25f }
        if (!hasCenterKnown && threeLanes.hasDepth) {
            val centerLane = threeLanes.lanes["center"]
            if (centerLane != null && (centerLane.closeRatio >= 0.42f || centerLane.medianM <= 0.95f)) {
                val box = EdgeBox((imgW * 0.33f).toInt(), (imgH * 0.22f).toInt(), (imgW * 0.67f).toInt(), (imgH * 0.78f).toInt())
                candidates.add(
                    EdgeCandidate(
                        className = "obstacle",
                        classNameVn = "vật cản",
                        confidence = 0.88f,
                        position = "center",
                        distanceM = centerLane.medianM.coerceIn(0.4f, 0.95f),
                        inRoi = true,
                        isDanger = true,
                        boundingBox = box,
                        sourceModel = "DepthV2-SafetyInversion",
                    )
                )
            }
        }

        val analysis = decisionEngine.processFrame(
            rawCandidates = candidates,
            depthLanes = threeLanes,
            imgW = imgW,
            imgH = imgH,
            yoloeCount = yoloeRaw.size,
            auxHazardCount = auxUsedCount,
            segHazardCount = segAddedCount,
            runtimeLabel = bundle.runtimeSummary,
        )

        var frameB64: String? = null
        var depthB64: String? = null
        if (needTelemetryVisuals) {
            frameB64 = encodeBitmapJpegB64(frameBitmap, quality = 58)
            if (dmap != null) {
                depthB64 = encodeDepthHeatmapB64(dmap, depthW, depthH)
            }
        }

        val totalMs = ((System.nanoTime() - t0) / 1_000_000.0f * 10f).toInt() / 10f
        return EdgeFramePacket(
            analysis = analysis,
            processingMs = totalMs,
            runtimeSummary = bundle.runtimeSummary,
            frameB64 = frameB64,
            depthB64 = depthB64,
        )
    }

    private fun prepareYoloLetterboxTensor(src: Bitmap, sz: Int): Triple<Float, Float, Float> {
        val srcW = src.width.toFloat()
        val srcH = src.height.toFloat()
        val scale = min(sz / srcW, sz / srcH)
        val newW = (srcW * scale).toInt().coerceAtLeast(1)
        val newH = (srcH * scale).toInt().coerceAtLeast(1)
        val padX = (sz - newW) * 0.5f
        val padY = (sz - newH) * 0.5f

        letterboxCanvas.drawColor(Color.rgb(114, 114, 114))
        val scaled = Bitmap.createScaledBitmap(src, newW, newH, true)
        letterboxCanvas.drawBitmap(scaled, padX, padY, null)
        if (scaled != src) scaled.recycle()

        letterboxBmp.getPixels(yoloPixels, 0, sz, 0, 0, sz, sz)
        val planeSize = sz * sz
        yoloBuf.clear()
        val inv255 = 1f / 255f
        for (i in 0 until planeSize) {
            val px = yoloPixels[i]
            yoloBuf.put(i, ((px ushr 16) and 0xFF) * inv255)
            yoloBuf.put(planeSize + i, ((px ushr 8) and 0xFF) * inv255)
            yoloBuf.put(planeSize * 2 + i, (px and 0xFF) * inv255)
        }
        yoloBuf.rewind()
        return Triple(scale, padX, padY)
    }

    private fun runYoloSession(
        session: EdgeModelsManager.LoadedSession,
        env: ai.onnxruntime.OrtEnvironment,
        names: List<String>,
        imgsz: Int,
        confThresh: Float,
        srcW: Int,
        srcH: Int,
        scale: Float,
        padX: Float,
        padY: Float,
    ): List<RawBoxDet> {
        if (names.isEmpty()) return emptyList()
        yoloBuf.rewind()
        val inputName = session.session.inputNames.firstOrNull() ?: "images"
        OnnxTensor.createTensor(env, yoloBuf, longArrayOf(1L, 3L, imgsz.toLong(), imgsz.toLong())).use { tensor ->
            session.session.run(mapOf(inputName to tensor)).use { result ->
                val outTensor = result[0] as OnnxTensor
                val shape = outTensor.info.shape
                if (shape.size < 3) return emptyList()
                val channels = shape[1].toInt()
                val anchors = shape[2].toInt()
                val numClasses = min(names.size, channels - 4)
                if (numClasses <= 0 || anchors <= 0) return emptyList()

                val fb = outTensor.floatBuffer
                val rawCands = ArrayList<RawBoxDet>(16)

                for (a in 0 until anchors) {
                    var bestCls = -1
                    var bestScore = confThresh
                    for (c in 0 until numClasses) {
                        val s = fb.get((4 + c) * anchors + a)
                        if (s > bestScore) {
                            bestScore = s
                            bestCls = c
                        }
                    }
                    if (bestCls < 0) continue

                    val cx = fb.get(a)
                    val cy = fb.get(anchors + a)
                    val bw = fb.get(2 * anchors + a)
                    val bh = fb.get(3 * anchors + a)

                    val x1 = (((cx - bw * 0.5f) - padX) / scale).toInt().coerceIn(0, srcW - 1)
                    val y1 = (((cy - bh * 0.5f) - padY) / scale).toInt().coerceIn(0, srcH - 1)
                    val x2 = (((cx + bw * 0.5f) - padX) / scale).toInt().coerceIn(x1 + 1, srcW)
                    val y2 = (((cy + bh * 0.5f) - padY) / scale).toInt().coerceIn(y1 + 1, srcH)

                    val rawName = names[bestCls]
                    val normName = DistanceEstimator.normalizeClass(rawName)
                    rawCands.add(RawBoxDet(bestCls, rawName, normName, bestScore, x1, y1, x2, y2))
                }
                return nonMaxSuppression(rawCands, iouThresh = 0.45f)
            }
        }
    }

    private fun nonMaxSuppression(dets: List<RawBoxDet>, iouThresh: Float): List<RawBoxDet> {
        if (dets.size <= 1) return dets
        val sorted = dets.sortedByDescending { it.confidence }
        val kept = ArrayList<RawBoxDet>(sorted.size)
        for (cand in sorted) {
            var suppr = false
            for (k in kept) {
                if (k.normName != cand.normName) continue
                val ix1 = max(cand.x1, k.x1)
                val iy1 = max(cand.y1, k.y1)
                val ix2 = min(cand.x2, k.x2)
                val iy2 = min(cand.y2, k.y2)
                if (ix2 > ix1 && iy2 > iy1) {
                    val inter = (ix2 - ix1).toFloat() * (iy2 - iy1).toFloat()
                    val areaA = max(1f, (cand.x2 - cand.x1).toFloat() * (cand.y2 - cand.y1).toFloat())
                    val areaB = max(1f, (k.x2 - k.x1).toFloat() * (k.y2 - k.y1).toFloat())
                    if (inter / max(1f, areaA + areaB - inter) >= iouThresh) {
                        suppr = true
                        break
                    }
                }
            }
            if (!suppr) kept.add(cand)
        }
        return kept
    }

    private fun runDepthEstimator(
        src: Bitmap,
        bundle: EdgeModelsManager.EdgeBundle,
        manifest: ModelsManifest,
        env: ai.onnxruntime.OrtEnvironment,
    ) {
        val sess = bundle.depth ?: return
        val sz = manifest.depthImgsz.coerceAtLeast(252)
        val scaled = Bitmap.createScaledBitmap(src, sz, sz, true)
        scaled.getPixels(depthPixels, 0, sz, 0, 0, sz, sz)
        if (scaled != src) scaled.recycle()

        val meanR = manifest.imagenetMean.getOrElse(0) { 0.485f }
        val meanG = manifest.imagenetMean.getOrElse(1) { 0.456f }
        val meanB = manifest.imagenetMean.getOrElse(2) { 0.406f }
        val invStdR = 1f / (255f * manifest.imagenetStd.getOrElse(0) { 0.229f })
        val invStdG = 1f / (255f * manifest.imagenetStd.getOrElse(1) { 0.224f })
        val invStdB = 1f / (255f * manifest.imagenetStd.getOrElse(2) { 0.225f })
        val invStd255R = 1f / manifest.imagenetStd.getOrElse(0) { 0.229f }
        val invStd255G = 1f / manifest.imagenetStd.getOrElse(1) { 0.224f }
        val invStd255B = 1f / manifest.imagenetStd.getOrElse(2) { 0.225f }

        val planeSize = sz * sz
        depthBuf.clear()
        for (i in 0 until planeSize) {
            val px = depthPixels[i]
            val r = ((px ushr 16) and 0xFF) / 255f
            val g = ((px ushr 8) and 0xFF) / 255f
            val b = (px and 0xFF) / 255f
            depthBuf.put(i, (r - meanR) * invStd255R)
            depthBuf.put(planeSize + i, (g - meanG) * invStd255G)
            depthBuf.put(planeSize * 2 + i, (b - meanB) * invStd255B)
        }
        depthBuf.rewind()

        val inName = sess.session.inputNames.firstOrNull() ?: "pixel_values"
        try {
            OnnxTensor.createTensor(env, depthBuf, longArrayOf(1L, 3L, sz.toLong(), sz.toLong())).use { t ->
                sess.session.run(mapOf(inName to t)).use { res ->
                    val outTensor = res[0] as OnnxTensor
                    val fb = outTensor.floatBuffer
                    val arr = FloatArray(sz * sz)
                    for (i in arr.indices) {
                        arr[i] = fb.get(i).coerceIn(0.2f, 15.0f)
                    }
                    latestDepthMap = arr
                    depthW = sz
                    depthH = sz
                }
            }
        } catch (e: Exception) {
            Log.w(tag, "Lỗi chạy DepthV2 trên thiết bị: ${e.message}")
        }
        // Giữ compiler không cảnh báo biến cục bộ chưa dùng nếu dọn sau
        if (invStdR + invStdG + invStdB < 0f) return
    }

    private fun runSegFormer(
        src: Bitmap,
        bundle: EdgeModelsManager.EdgeBundle,
        manifest: ModelsManifest,
        env: ai.onnxruntime.OrtEnvironment,
    ) {
        val sess = bundle.segformer ?: return
        val sz = manifest.segImgsz.coerceAtLeast(256)
        val scaled = Bitmap.createScaledBitmap(src, sz, sz, true)
        scaled.getPixels(segPixels, 0, sz, 0, 0, sz, sz)
        if (scaled != src) scaled.recycle()

        val meanR = manifest.imagenetMean.getOrElse(0) { 0.485f }
        val meanG = manifest.imagenetMean.getOrElse(1) { 0.456f }
        val meanB = manifest.imagenetMean.getOrElse(2) { 0.406f }
        val invStdR = 1f / manifest.imagenetStd.getOrElse(0) { 0.229f }
        val invStdG = 1f / manifest.imagenetStd.getOrElse(1) { 0.224f }
        val invStdB = 1f / manifest.imagenetStd.getOrElse(2) { 0.225f }

        val planeSize = sz * sz
        segBuf.clear()
        for (i in 0 until planeSize) {
            val px = segPixels[i]
            segBuf.put(i, (((px ushr 16) and 0xFF) / 255f - meanR) * invStdR)
            segBuf.put(planeSize + i, (((px ushr 8) and 0xFF) / 255f - meanG) * invStdG)
            segBuf.put(planeSize * 2 + i, ((px and 0xFF) / 255f - meanB) * invStdB)
        }
        segBuf.rewind()

        val inName = sess.session.inputNames.firstOrNull() ?: "pixel_values"
        try {
            OnnxTensor.createTensor(env, segBuf, longArrayOf(1L, 3L, sz.toLong(), sz.toLong())).use { t ->
                sess.session.run(mapOf(inName to t)).use { res ->
                    val outTensor = res[0] as OnnxTensor
                    val shape = outTensor.info.shape // [1, 150, 64, 64]
                    if (shape.size < 4) return
                    val numCls = shape[1].toInt()
                    val gh = shape[2].toInt()
                    val gw = shape[3].toInt()
                    val cells = gh * gw
                    val fb = outTensor.floatBuffer

                    val argmax = IntArray(cells)
                    val counts = IntArray(numCls)
                    val minX = IntArray(numCls) { gw }
                    val minY = IntArray(numCls) { gh }
                    val maxX = IntArray(numCls) { -1 }
                    val maxY = IntArray(numCls) { -1 }

                    for (cell in 0 until cells) {
                        var bestC = 0
                        var bestV = Float.NEGATIVE_INFINITY
                        for (c in 0 until numCls) {
                            val v = fb.get(c * cells + cell)
                            if (v > bestV) {
                                bestV = v
                                bestC = c
                            }
                        }
                        argmax[cell] = bestC
                        counts[bestC]++
                        val x = cell % gw
                        val y = cell / gw
                        if (x < minX[bestC]) minX[bestC] = x
                        if (y < minY[bestC]) minY[bestC] = y
                        if (x > maxX[bestC]) maxX[bestC] = x
                        if (y > maxY[bestC]) maxY[bestC] = y
                    }
                    latestSegIds64 = argmax

                    val extracted = ArrayList<SegNormHazard>()
                    val total = cells.toFloat()
                    for (c in 0 until numCls) {
                        val pixRatio = counts[c] / total
                        if (pixRatio < 0.05f) continue
                        val rawLabel = manifest.segformerId2Label[c.toString()]?.lowercase()?.trim() ?: continue
                        val mapped = DistanceEstimator.SEG_OBSTACLE_MAP[rawLabel] ?: continue
                        val (clsEn, clsVn) = mapped
                        if (clsEn == "wall") continue // Tường sẽ được DepthV2 Safety Inversion kiểm soát theo ngưỡng <= 1.0m
                        val x1n = minX[c] / gw.toFloat()
                        val y1n = minY[c] / gh.toFloat()
                        val x2n = (maxX[c] + 1) / gw.toFloat()
                        val y2n = (maxY[c] + 1) / gh.toFloat()
                        extracted.add(SegNormHazard(clsEn, clsVn, x1n, y1n, x2n, y2n, 1.8f))
                    }
                    latestSegHazardsNorm = extracted
                }
            }
        } catch (e: Exception) {
            Log.w(tag, "Lỗi chạy SegFormer trên thiết bị: ${e.message}")
        }
    }

    private fun isValidPotholeRegion(box: EdgeBox, imgW: Int, imgH: Int, id2label: Map<String, String>): Boolean {
        if (box.y2 <= imgH * 0.34f) return false
        val seg = latestSegIds64 ?: return true
        val rx1 = (box.x1 * segSw / max(1, imgW)).coerceIn(0, segSw - 1)
        val ry1 = (box.y1 * segSh / max(1, imgH)).coerceIn(0, segSh - 1)
        val rx2 = (box.x2 * segSw / max(1, imgW)).coerceIn(rx1 + 1, segSw)
        val ry2 = (box.y2 * segSh / max(1, imgH)).coerceIn(ry1 + 1, segSh)
        var total = 0
        var groundCnt = 0
        var nonGroundCnt = 0
        for (y in ry1 until ry2) {
            val off = y * segSw
            for (x in rx1 until rx2) {
                val cid = seg[off + x]
                val lbl = id2label[cid.toString()]?.lowercase()?.trim() ?: ""
                total++
                if (lbl in DistanceEstimator.POTHOLE_GROUND_LABELS) groundCnt++
                if (lbl in DistanceEstimator.POTHOLE_NON_GROUND_LABELS) nonGroundCnt++
            }
        }
        if (total == 0) return true
        if (nonGroundCnt.toFloat() / total > 0.35f) return false
        return (groundCnt.toFloat() / total) >= 0.25f || nonGroundCnt == 0
    }

    private fun isCoveredByBoxes(target: EdgeBox, known: List<EdgeBox>, thresh: Float): Boolean {
        val area = max(1f, target.width.toFloat() * target.height.toFloat())
        return known.any { b ->
            val ix1 = max(target.x1, b.x1)
            val iy1 = max(target.y1, b.y1)
            val ix2 = min(target.x2, b.x2)
            val iy2 = min(target.y2, b.y2)
            ix2 > ix1 && iy2 > iy1 && (((ix2 - ix1).toFloat() * (iy2 - iy1).toFloat()) / area) > thresh
        }
    }

    private fun encodeBitmapJpegB64(bmp: Bitmap, quality: Int): String {
        val targetW = 360
        val targetH = ((bmp.height.toFloat() / max(1, bmp.width)) * targetW).toInt().coerceAtLeast(120)
        val small = Bitmap.createScaledBitmap(bmp, targetW, targetH, true)
        val baos = ByteArrayOutputStream()
        small.compress(Bitmap.CompressFormat.JPEG, quality, baos)
        if (small != bmp) small.recycle()
        return Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)
    }

    private fun encodeDepthHeatmapB64(dmap: FloatArray, dw: Int, dh: Int): String {
        val outW = 126
        val outH = 126
        val pixels = IntArray(outW * outH)
        for (y in 0 until outH) {
            val sy = (y * dh) / outH
            for (x in 0 until outW) {
                val sx = (x * dw) / outW
                val dist = dmap[sy * dw + sx]
                // Đỏ = rất gần (<= 0.8m), Vàng/Xanh lá = trung bình (~2.2m), Xanh dương = xa (>= 4.5m)
                val t = ((dist - 0.5f) / 4.0f).coerceIn(0f, 1f)
                val r = ((1f - t) * 255f).toInt().coerceIn(0, 255)
                val g = ((1f - kotlin.math.abs(t - 0.5f) * 2f) * 255f).toInt().coerceIn(0, 255)
                val b = (t * 255f).toInt().coerceIn(0, 255)
                pixels[y * outW + x] = Color.rgb(r, g, b)
            }
        }
        val bmp = Bitmap.createBitmap(pixels, outW, outH, Bitmap.Config.ARGB_8888)
        val baos = ByteArrayOutputStream()
        bmp.compress(Bitmap.CompressFormat.JPEG, 55, baos)
        bmp.recycle()
        return Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)
    }
}
