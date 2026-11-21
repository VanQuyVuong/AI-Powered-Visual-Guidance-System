package com.example.ai_hkdn.edge

import com.example.ai_hkdn.data.model.BoundingBox
import com.example.ai_hkdn.data.model.DetectedObject
import com.example.ai_hkdn.data.model.GuidanceInfo
import com.example.ai_hkdn.data.model.NavigationInfo
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Kết quả sau khi hợp nhất đa mô hình + đánh giá lối đi + chống lặp giọng nói trên thiết bị.
 */
data class EdgeAnalysisOutput(
    val detectedObjects: List<DetectedObject>,
    val guidance: GuidanceInfo,
    val displayText: String,
    val navigation: NavigationInfo,
    val stateCode: String,
    val stateLabel: String,
    val spokenThisFrame: Boolean,
    val instantAction: String,
    val threeLaneDepth: ThreeLaneDepth,
    val fusionSummaryMap: Map<String, Any>,
    val decisionTelemetryMap: Map<String, Any>,
)

/**
 * Bộ Tổng Hợp & Máy Trạng Thái Dẫn Đường Chạy Độc Lập Trên Android (Kotlin):
 * Hợp nhất YOLOE-26s + ObstacleYOLO11n + DepthV2 + SegFormer-B0 theo đúng logic
 * của `fusion_engine.py` và `guided_scan.py` trên server Python.
 */
class EdgeDecisionEngine(
    private val emaAlpha: Float = 0.68f,
    private val memoryTtlMs: Long = 1600L,
) {
    companion object {
        const val WALL_DANGER_M = 1.00f
        const val OBSTACLE_DANGER_M = 2.25f
        const val SIDE_OBSTACLE_M = 1.65f
        const val ALERT_REPEAT_MS = 6000L
        const val CLEAR_REPEAT_MS = 6500L

        val GROUND_HAZARD_CLASSES = setOf(
            "stairs", "step", "curb", "doorstep", "threshold", "pothole", "dropoff", "open_manhole"
        )
        val WALL_CLASSES = setOf("wall")

        val SEMANTIC_PRIORITY = mapOf(
            "stairs" to 100,
            "open_manhole" to 98,
            "dropoff" to 96,
            "step" to 95,
            "pothole" to 95,
            "person" to 90,
            "car" to 88,
            "mirror" to 86,
            "glass_door" to 86,
            "column" to 85,
            "pillar" to 85,
            "motorcycle" to 85,
            "bicycle" to 85,
            "door" to 82,
            "chair" to 80,
            "table" to 80,
            "dining table" to 80,
            "desk" to 80,
            "cabinet" to 78,
            "couch" to 78,
            "bed" to 78,
            "refrigerator" to 78,
            "wall" to 40,
            "obstacle" to 30,
        )
    }

    private data class TrackEntry(val timestampMs: Long, val smoothedDistM: Float)

    private val temporalTracks = HashMap<String, TrackEntry>()

    // Trạng thái máy dẫn đường Guided Scan
    private var lastText: String = ""
    private var lastPriority: String = "LOW"
    private var lastSpeechTimeMs: Long = 0L
    private var lastAlertSig: String = ""
    private var lastDistM: Float = 9.9f
    private var clearFrames: Int = 0
    private var blockedFrames: Int = 0
    private var pendingPostTurnReconfirm: Boolean = false
    private var currentStatus: String = "CLEAR"
    private var lastSpokenFlag: Boolean = false

    @Synchronized
    fun reset() {
        temporalTracks.clear()
        lastText = ""
        lastPriority = "LOW"
        lastSpeechTimeMs = 0L
        lastAlertSig = ""
        lastDistM = 9.9f
        clearFrames = 0
        blockedFrames = 0
        pendingPostTurnReconfirm = false
        currentStatus = "CLEAR"
        lastSpokenFlag = false
    }

    @Synchronized
    fun processFrame(
        rawCandidates: List<EdgeCandidate>,
        depthLanes: ThreeLaneDepth,
        imgW: Int,
        imgH: Int,
        yoloeCount: Int,
        auxHazardCount: Int,
        segHazardCount: Int,
        runtimeLabel: String,
        nowMs: Long = System.currentTimeMillis(),
    ): EdgeAnalysisOutput {
        val t0 = System.nanoTime()

        // 1. Dọn track quá hạn TTL
        val it = temporalTracks.entries.iterator()
        while (it.hasNext()) {
            if ((nowMs - it.next().value.timestampMs) > memoryTtlMs) {
                it.remove()
            }
        }

        // 2. Lọc vật thể ảo phản chiếu trong gương TRƯỚC khi deduplication ưu tiên ngữ nghĩa
        val mirrorCleaned = filterReflectionsInsideMirrors(rawCandidates)

        // 3. Sắp xếp ưu tiên ngữ nghĩa -> khoảng cách gần nhất -> khử trùng IoU + Containment (>= 0.35)
        val sorted = mirrorCleaned.sortedWith(
            compareBy<EdgeCandidate>(
                { -(SEMANTIC_PRIORITY[it.className.lowercase()] ?: 60) },
                { it.distanceM }
            )
        )

        val fused = ArrayList<EdgeCandidate>(sorted.size)
        var hasSpecificCenterHazard = false

        for (cand in sorted) {
            val cLower = cand.className.lowercase()
            val cPrio = SEMANTIC_PRIORITY[cLower] ?: 60
            if ((cLower == "wall" || cLower == "obstacle") && hasSpecificCenterHazard) {
                continue
            }

            val boxA = cand.boundingBox
            val areaA = max(1f, boxA.width.toFloat() * boxA.height.toFloat())
            var isDup = false
            for (kept in fused) {
                val kPrio = SEMANTIC_PRIORITY[kept.className.lowercase()] ?: 60
                if (kPrio < cPrio) continue
                val boxB = kept.boundingBox
                if (boxA.x2 <= boxB.x1 || boxB.x2 <= boxA.x1 || boxA.y2 <= boxB.y1 || boxB.y2 <= boxA.y1) {
                    continue
                }
                val inter = (min(boxA.x2, boxB.x2) - max(boxA.x1, boxB.x1)).toFloat() *
                    (min(boxA.y2, boxB.y2) - max(boxA.y1, boxB.y1)).toFloat()
                val areaB = max(1f, boxB.width.toFloat() * boxB.height.toFloat())
                val union = areaA + areaB - inter
                val overlap = max(inter / max(1f, union), (inter / min(areaA, areaB)) * 0.65f)
                if (overlap >= 0.35f) {
                    isDup = true
                    break
                }
            }
            if (isDup) continue

            val smoothedD = smoothDistance(cLower, cand.position, cand.distanceM, nowMs)
            val isDanger = when (cLower) {
                "wall" -> smoothedD <= 1.00f
                "stairs", "step", "dropoff", "pothole", "open_manhole" -> smoothedD <= 2.40f
                "column", "pillar", "mirror", "glass_door" -> smoothedD <= 1.85f
                "obstacle" -> smoothedD <= 1.50f
                else -> (cand.inRoi || cand.position == "center") && smoothedD <= 2.25f || smoothedD <= 1.65f
            }

            if (cPrio >= 75 && cand.position == "center" && isDanger) {
                hasSpecificCenterHazard = true
            }
            fused.add(cand.copy(distanceM = smoothedD, isDanger = isDanger))
        }

        val filtered = fused

        // 4. Tổng hợp khoảng trống 3 làn
        val laneSummary = HashMap<String, Map<String, Any?>>()
        val laneBlockedMap = HashMap<String, Boolean>()
        val laneClearDist = HashMap<String, Float>()
        for (laneName in listOf("left", "center", "right")) {
            val dInfo = depthLanes.lanes[laneName]
            val medM = dInfo?.medianM ?: 3.5f
            val laneObjs = filtered.filter { it.position == laneName && it.isDanger }
            val nearest = laneObjs.minByOrNull { it.distanceM }
            val effClear = DistanceEstimator.round2(min(medM, nearest?.distanceM ?: medM))
            val blk = laneObjs.isNotEmpty() || ((dInfo?.isBlocked == true) && medM <= 1.00f)
            laneBlockedMap[laneName] = blk
            laneClearDist[laneName] = effClear
            laneSummary[laneName] = mapOf(
                "clear_distance_m" to effClear,
                "is_blocked" to blk,
                "primary_hazard" to nearest?.classNameVn,
            )
        }

        val instantAction = when {
            laneBlockedMap["center"] != true -> "GO_STRAIGHT"
            laneBlockedMap["left"] != true && laneBlockedMap["right"] != true ->
                if ((laneClearDist["left"] ?: 0f) >= (laneClearDist["right"] ?: 0f)) "TURN_LEFT" else "TURN_RIGHT"
            laneBlockedMap["left"] != true -> "TURN_LEFT"
            laneBlockedMap["right"] != true -> "TURN_RIGHT"
            else -> "TURN_AROUND"
        }

        // 5. Máy trạng thái Guided Scan + chống lặp giọng nói
        val assessment = assessPath(filtered, depthLanes, imgW)
        var status = assessment.first
        val hazardName = assessment.second
        val minDist = assessment.third

        val isClearLike = status == "CLEAR" || status == "ALLEY_CLEAR"
        clearFrames = if (isClearLike) clearFrames + 1 else 0
        blockedFrames = if (!isClearLike) blockedFrames + 1 else 0

        val sinceMs = if (lastSpeechTimeMs > 0L) nowMs - lastSpeechTimeMs else Long.MAX_VALUE
        val wasTurningOrBlocked = lastAlertSig.isNotEmpty() && lastAlertSig != "CLEAR" && lastAlertSig != "ALLEY_CLEAR"
        if (status == "ALLEY_CLEAR" && lastAlertSig.startsWith("TURN_")) {
            status = "CLEAR"
        }

        var sig = "$status:$hazardName"
        val targetText: String
        val targetPrio: String
        when (status) {
            "STEPS" -> {
                if (hazardName == "hố sâu") {
                    targetText = "Chú ý dưới chân, có hố sâu, hãy dừng lại."
                    targetPrio = "CRITICAL"
                } else {
                    targetText = "Chú ý dưới chân, có $hazardName, hãy bước chậm."
                    targetPrio = "HIGH"
                }
            }
            "TURN_AROUND" -> {
                targetText = "Phía trước có $hazardName, cả hai bên đều vướng, hãy quay đầu."
                targetPrio = "CRITICAL"
            }
            "TURN_LEFT" -> {
                targetText = "Phía trước có $hazardName, hãy rẽ sang trái."
                targetPrio = "HIGH"
            }
            "TURN_RIGHT" -> {
                targetText = "Phía trước có $hazardName, hãy rẽ sang phải."
                targetPrio = "HIGH"
            }
            "ALLEY_DRIFT_LEFT" -> {
                targetText = "Đang đi sát tường bên trái, hãy chỉnh nhẹ sang phải rồi đi thẳng."
                targetPrio = "HIGH"
                sig = "ALLEY_DRIFT_LEFT"
            }
            "ALLEY_DRIFT_RIGHT" -> {
                targetText = "Đang đi sát tường bên phải, hãy chỉnh nhẹ sang trái rồi đi thẳng."
                targetPrio = "HIGH"
                sig = "ALLEY_DRIFT_RIGHT"
            }
            "ALLEY_CLEAR" -> {
                targetText = "Hai bên là bức tường dài, phía trước thông thoáng, bạn hãy đi thẳng."
                targetPrio = "LOW"
                sig = "ALLEY_CLEAR"
            }
            else -> {
                targetText = "Phía trước an toàn, đi thẳng."
                targetPrio = "LOW"
                sig = "CLEAR"
            }
        }

        var shouldSpeak = false
        val isFreshFromClear = lastAlertSig == "CLEAR" || lastAlertSig == "ALLEY_CLEAR" || lastAlertSig.isEmpty()

        when {
            status == "STEPS" -> {
                val minStepMs = if (isFreshFromClear) 0L else 1200L
                if ((lastAlertSig != sig && sinceMs >= minStepMs) || sinceMs >= ALERT_REPEAT_MS) {
                    shouldSpeak = true
                    pendingPostTurnReconfirm = false
                }
            }
            status in setOf("TURN_LEFT", "TURN_RIGHT", "TURN_AROUND", "ALLEY_DRIFT_LEFT", "ALLEY_DRIFT_RIGHT") -> {
                val urgentReroute = (lastAlertSig != sig) && (minDist <= 1.15f)
                val minF = if (isFreshFromClear || urgentReroute) 1 else 2
                val minIntMs = if (isFreshFromClear || urgentReroute) 0L else 900L
                val criticalApproach = (minDist <= 0.85f && (lastDistM - minDist) >= 0.45f && sinceMs >= 1800L)
                if ((lastAlertSig != sig && blockedFrames >= minF && sinceMs >= minIntMs) ||
                    criticalApproach ||
                    sinceMs >= ALERT_REPEAT_MS
                ) {
                    shouldSpeak = true
                    pendingPostTurnReconfirm = false
                }
            }
            status == "ALLEY_CLEAR" -> {
                val minAlleyMs = if (lastAlertSig.isEmpty()) 0L else 600L
                if (lastAlertSig != "ALLEY_CLEAR" && sinceMs >= minAlleyMs) {
                    shouldSpeak = true
                    pendingPostTurnReconfirm = false
                }
            }
            else -> {
                val minClearMs = if (wasTurningOrBlocked) 550L else 1500L
                if (clearFrames >= 2 && sinceMs >= minClearMs && (lastAlertSig != "CLEAR" || sinceMs >= CLEAR_REPEAT_MS)) {
                    shouldSpeak = true
                    pendingPostTurnReconfirm = wasTurningOrBlocked
                } else if (clearFrames >= 2 && pendingPostTurnReconfirm && sinceMs >= 2400L) {
                    shouldSpeak = true
                    pendingPostTurnReconfirm = false
                }
            }
        }

        currentStatus = status
        lastSpokenFlag = shouldSpeak
        if (shouldSpeak) {
            lastText = targetText
            lastPriority = targetPrio
            lastSpeechTimeMs = nowMs
            lastAlertSig = sig
            lastDistM = minDist
        }

        val displayText = if (shouldSpeak) targetText else (lastText.ifEmpty { targetText })
        val speechText = if (shouldSpeak) targetText else ""
        val effectivePrio = if (shouldSpeak) targetPrio else (if (lastText.isNotEmpty()) lastPriority else targetPrio)

        val navMapping = when (status) {
            "TURN_LEFT" -> "left" to "Rẽ trái"
            "ALLEY_DRIFT_RIGHT" -> "left" to "Chỉnh nhẹ sang trái"
            "TURN_RIGHT" -> "right" to "Rẽ phải"
            "ALLEY_DRIFT_LEFT" -> "right" to "Chỉnh nhẹ sang phải"
            "TURN_AROUND" -> "stop" to "Dừng lại / Quay đầu"
            "STEPS" -> "stop" to "Bước chậm / Chú ý dưới chân"
            "ALLEY_CLEAR" -> "straight" to "Đi thẳng trong hẻm"
            else -> "straight" to "Đi thẳng"
        }

        val uiDetections = filtered.mapIndexed { idx, obj ->
            DetectedObject(
                id = idx + 1,
                className = obj.className,
                classNameVn = obj.classNameVn,
                confidence = obj.confidence,
                position = obj.position,
                distanceM = obj.distanceM,
                isDanger = obj.isDanger,
                boundingBox = BoundingBox(obj.boundingBox.x1, obj.boundingBox.y1, obj.boundingBox.x2, obj.boundingBox.y2),
            )
        }

        val fusionLatencyMs = ((System.nanoTime() - t0) / 1_000_000.0 * 100.0).toInt() / 100.0
        val centerL = depthLanes.lanes["center"]?.medianM ?: 3.5f
        val leftL = depthLanes.lanes["left"]?.medianM ?: 3.5f
        val rightL = depthLanes.lanes["right"]?.medianM ?: 3.5f
        val stateLabel = mapOf(
            "CLEAR" to "ĐI THẲNG AN TOÀN (CLEAR)",
            "ALLEY_CLEAR" to "ĐI HẺM TĨNH LẶNG (ALLEY_CLEAR)",
            "ALLEY_DRIFT_LEFT" to "LỆCH TƯỜNG TRÁI",
            "ALLEY_DRIFT_RIGHT" to "LỆCH TƯỜNG PHẢI",
            "TURN_LEFT" to "RẼ TRÁI (TURN_LEFT)",
            "TURN_RIGHT" to "RẼ PHẢI (TURN_RIGHT)",
            "TURN_AROUND" to "QUAY ĐẦU (TURN_AROUND)",
            "STEPS" to "CẢNH BÁO DƯỚI CHÂN (STEPS)",
        )[status] ?: status

        val sinceSec = if (lastSpeechTimeMs > 0L) max(0.0, (nowMs - lastSpeechTimeMs) / 1000.0) else 0.0
        val decisionTelemetryMap = mapOf<String, Any>(
            "state_code" to status,
            "state_label" to stateLabel,
            "clear_frames" to clearFrames,
            "blocked_frames" to blockedFrames,
            "since_last_speech_s" to ((sinceSec * 10.0).toInt() / 10.0),
            "last_speech_ts" to (lastSpeechTimeMs / 1000.0),
            "spoken_this_frame" to shouldSpeak,
            "voice_engine" to "Edge $runtimeLabel + Android TTS",
        )

        val dangerCount = filtered.count { it.isDanger }
        val fusionSummaryMap = mapOf<String, Any>(
            "active_models" to listOf("YOLOE-26s-FP16", "ObstacleYOLO11n-FP16", "DepthV2-FP16", "SegFormer-B0-FP16"),
            "raw_candidate_count" to rawCandidates.size,
            "fused_obstacle_count" to filtered.size,
            "danger_obstacle_count" to dangerCount,
            "dedup_removed_count" to max(0, rawCandidates.size - filtered.size),
            "ema_active_tracks" to temporalTracks.size,
            "fusion_latency_ms" to fusionLatencyMs,
            "instant_action" to instantAction,
            "lane_clearance" to laneSummary,
            "decision_telemetry" to decisionTelemetryMap,
            "model_branches" to mapOf(
                "yolo11" to mapOf(
                    "model_name" to "YOLOE-26s + Obstacle11n (NPU)",
                    "count" to (yoloeCount + auxHazardCount),
                    "detail" to (filtered.take(3).joinToString(", ") { it.classNameVn }.ifEmpty { "Không có vật cản định danh" }),
                ),
                "depth_v2" to mapOf(
                    "model_name" to "DepthV2 Metric FP16",
                    "has_depth" to depthLanes.hasDepth,
                    "center_m" to DistanceEstimator.round2(centerL),
                    "left_m" to DistanceEstimator.round2(leftL),
                    "right_m" to DistanceEstimator.round2(rightL),
                    "detail" to String.format(java.util.Locale.US, "T:%.1fm | G:%.1fm | P:%.1fm", leftL, centerL, rightL),
                ),
                "edgesam" to mapOf(
                    "model_name" to runtimeLabel,
                    "active_masks" to yoloeCount,
                    "detail" to "Chạy trực tiếp trên chip điện thoại",
                ),
                "segformer_geo" to mapOf(
                    "model_name" to "SegFormer-B0 FP16",
                    "count" to segHazardCount,
                    "detail" to if (segHazardCount > 0) "Phát hiện $segHazardCount cấu trúc" else "Mặt sàn thông thoáng",
                ),
            ),
        )

        return EdgeAnalysisOutput(
            detectedObjects = uiDetections,
            guidance = GuidanceInfo(text = speechText, priority = effectivePrio),
            displayText = displayText,
            navigation = NavigationInfo(direction = navMapping.first, confidence = 0.92f, text = navMapping.second),
            stateCode = status,
            stateLabel = stateLabel,
            spokenThisFrame = shouldSpeak,
            instantAction = instantAction,
            threeLaneDepth = depthLanes,
            fusionSummaryMap = fusionSummaryMap,
            decisionTelemetryMap = decisionTelemetryMap,
        )
    }

    private fun smoothDistance(className: String, position: String, rawDist: Float, nowMs: Long): Float {
        val key = "$className:$position"
        val prev = temporalTracks[key]
        val smoothed = if (prev != null && (nowMs - prev.timestampMs) <= memoryTtlMs) {
            val alpha = if (rawDist < prev.smoothedDistM) 0.82f else emaAlpha
            DistanceEstimator.round1(alpha * rawDist + (1f - alpha) * prev.smoothedDistM)
        } else {
            DistanceEstimator.round1(rawDist)
        }
        temporalTracks[key] = TrackEntry(nowMs, smoothed)
        return smoothed
    }

    private fun filterReflectionsInsideMirrors(objects: List<EdgeCandidate>): List<EdgeCandidate> {
        val mirrorBoxes = objects.filter { it.className.equals("mirror", ignoreCase = true) }.map { it.boundingBox }
        if (mirrorBoxes.isEmpty()) return objects
        return objects.filter { obj ->
            if (obj.className.equals("mirror", ignoreCase = true)) return@filter true
            val b = obj.boundingBox
            val area = max(1f, b.width.toFloat() * b.height.toFloat())
            val inside = mirrorBoxes.any { mb ->
                val ix1 = max(b.x1, mb.x1)
                val iy1 = max(b.y1, mb.y1)
                val ix2 = min(b.x2, mb.x2)
                val iy2 = min(b.y2, mb.y2)
                ix2 > ix1 && iy2 > iy1 && (((ix2 - ix1).toFloat() * (iy2 - iy1).toFloat()) / area) >= 0.55f
            }
            !inside
        }
    }

    private fun assessPath(
        objs: List<EdgeCandidate>,
        depthLanes: ThreeLaneDepth,
        imgW: Int,
    ): Triple<String, String, Float> {
        val w = max(1, imgW).toFloat()
        val hasDepth = depthLanes.hasDepth
        val l = depthLanes.lanes["left"]
        val c = depthLanes.lanes["center"]
        val r = depthLanes.lanes["right"]
        val lBlk = l?.isBlocked ?: false
        val cBlk = c?.isBlocked ?: false
        val rBlk = r?.isBlocked ?: false
        val lMed = l?.medianM ?: 3.0f
        val cMed = c?.medianM ?: 3.0f
        val rMed = r?.medianM ?: 3.0f

        fun isInPath(o: EdgeCandidate): Boolean {
            if (o.className.lowercase() in WALL_CLASSES) return false
            if (o.distanceM > OBSTACLE_DANGER_M) return false
            if (o.position == "left" || o.position == "right") {
                return (o.boundingBox.x1 < 0.58f * w) && (o.boundingBox.x2 > 0.42f * w)
            }
            if (o.position == "center") return true
            return (o.boundingBox.x1 < 0.62f * w && o.boundingBox.x2 > 0.38f * w) || o.inRoi
        }

        // 1. Nguy hiểm dưới chân (cầu thang / bậc thềm / ổ gà / hố ga / hố sâu)
        val stepObjs = objs.filter { it.className.lowercase() in GROUND_HAZARD_CLASSES && it.distanceM <= 2.40f }
        if (stepObjs.isNotEmpty()) {
            val nearest = stepObjs.minBy { it.distanceM }
            return Triple("STEPS", nearest.classNameVn, nearest.distanceM)
        }

        // 2. Bức tường ở gần trung tâm <= 1.0m
        val wallObjs = objs.filter {
            it.className.lowercase() in WALL_CLASSES && it.position == "center" && it.distanceM <= WALL_DANGER_M
        }
        val isWallNear = wallObjs.isNotEmpty()

        // 3. Vật cản trung tâm
        val centerObjs = objs.filter { isInPath(it) }
        val onlyFarObjs = objs.isNotEmpty() && centerObjs.isEmpty() && objs.all { it.distanceM > OBSTACLE_DANGER_M }
        val depthBlocked = cBlk && (if (onlyFarObjs) cMed <= 0.95f else cMed <= 1.00f)
        val centerBlocked = isWallNear || centerObjs.isNotEmpty() || depthBlocked

        if (!centerBlocked) {
            val sideNonWall = objs.any {
                it.position in setOf("left", "right") && it.className.lowercase() !in WALL_CLASSES && it.distanceM <= SIDE_OBSTACLE_M
            }
            val leftWall = objs.any { it.className.lowercase() in WALL_CLASSES && it.position == "left" && it.distanceM <= 1.65f }
            val rightWall = objs.any { it.className.lowercase() in WALL_CLASSES && it.position == "right" && it.distanceM <= 1.65f }
            val isAlley = !sideNonWall && (
                (hasDepth && lMed <= 1.45f && rMed <= 1.45f && cMed >= 1.95f) || (leftWall && rightWall)
                )
            if (isAlley) {
                if (hasDepth && lMed <= 0.72f && rMed > 0.80f) return Triple("ALLEY_DRIFT_LEFT", "tường bên trái", lMed)
                if (hasDepth && rMed <= 0.72f && lMed > 0.80f) return Triple("ALLEY_DRIFT_RIGHT", "tường bên phải", rMed)
                return Triple("ALLEY_CLEAR", "tường dài hai bên", cMed)
            }
            return Triple("CLEAR", "an toàn", 9.9f)
        }

        var minD = cMed
        val name: String = when {
            centerObjs.size >= 2 -> {
                minD = centerObjs.minOf { it.distanceM }
                val uniq = centerObjs.map { it.classNameVn }.toSet()
                if (uniq.size == 1 && uniq.first() != "vật cản") "nhiều ${uniq.first()}" else "nhiều vật cản"
            }
            centerObjs.size == 1 -> {
                val n = centerObjs.first()
                minD = n.distanceM
                n.classNameVn
            }
            isWallNear -> {
                minD = wallObjs.minOf { it.distanceM }
                "bức tường"
            }
            else -> "vật cản"
        }

        val leftNear = objs.any { it.position == "left" && it.distanceM <= SIDE_OBSTACLE_M && it !in centerObjs }
        val rightNear = objs.any { it.position == "right" && it.distanceM <= SIDE_OBSTACLE_M && it !in centerObjs }

        val leftGap: Float
        val rightGap: Float
        val wideObstacle: Boolean
        if (centerObjs.isNotEmpty()) {
            val ux1 = centerObjs.minOf { it.boundingBox.x1 }.toFloat()
            val ux2 = centerObjs.maxOf { it.boundingBox.x2 }.toFloat()
            leftGap = ux1 / w
            rightGap = (w - ux2) / w
            wideObstacle = (ux2 - ux1) / w >= 0.72f
        } else {
            leftGap = 0.33f
            rightGap = 0.33f
            wideObstacle = false
        }

        var leftBlocked = leftNear || lBlk || (leftGap < 0.18f) || (hasDepth && lMed <= 1.25f)
        var rightBlocked = rightNear || rBlk || (rightGap < 0.18f) || (hasDepth && rMed <= 1.25f)

        if (isWallNear) {
            if (!(hasDepth && lMed >= 1.9f)) leftBlocked = true
            if (!(hasDepth && rMed >= 1.9f)) rightBlocked = true
        } else if (wideObstacle || (hasDepth && lBlk && rBlk)) {
            leftBlocked = true
            rightBlocked = true
        }

        return when {
            leftBlocked && rightBlocked -> Triple("TURN_AROUND", name, minD)
            !leftBlocked && rightBlocked -> Triple("TURN_LEFT", name, minD)
            leftBlocked && !rightBlocked -> Triple("TURN_RIGHT", name, minD)
            else -> {
                val dirStatus = if (abs(leftGap - rightGap) >= 0.06f) {
                    if (leftGap > rightGap) "TURN_LEFT" else "TURN_RIGHT"
                } else {
                    if (lMed >= rMed) "TURN_LEFT" else "TURN_RIGHT"
                }
                Triple(dirStatus, name, minD)
            }
        }
    }
}
