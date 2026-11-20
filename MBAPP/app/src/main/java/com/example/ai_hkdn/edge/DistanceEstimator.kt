package com.example.ai_hkdn.edge

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Hộp bao vật thể nội bộ dùng cho bộ xử lý Edge AI trên điện thoại.
 */
data class EdgeBox(
    val x1: Int,
    val y1: Int,
    val x2: Int,
    val y2: Int,
) {
    val width: Int get() = max(0, x2 - x1)
    val height: Int get() = max(0, y2 - y1)
}

data class EdgeCandidate(
    val className: String,
    val classNameVn: String,
    val confidence: Float,
    val position: String,
    val distanceM: Float,
    val inRoi: Boolean,
    val isDanger: Boolean,
    val boundingBox: EdgeBox,
    val sourceModel: String,
)

data class LaneDepthState(
    val medianM: Float,
    val closeRatio: Float,
    val midRatio: Float,
    val isBlocked: Boolean,
)

data class ThreeLaneDepth(
    val hasDepth: Boolean,
    val lanes: Map<String, LaneDepthState>,
)

/**
 * Bộ hiệu chuẩn khoảng cách Pinhole + Depth Anything V2 + SegFormer ADE20K chạy trực tiếp bằng Kotlin.
 * Khớp 1:1 với `backend/app/vision/distance_estimator.py`, `pipeline.py` và `depth_service.py`.
 */
object DistanceEstimator {

    const val CAMERA_HEIGHT_M = 1.35f
    const val PERSON_HEIGHT_M = 1.70f
    const val PERSON_SHOULDER_WIDTH_M = 0.45f
    const val MAX_DISTANCE_M = 15.0f

    private const val TAN_HALF_FOV_Y_PORTRAIT = 0.625f
    private const val TAN_HALF_FOV_Y_LANDSCAPE = 0.477f
    private const val TAN_HALF_FOV_X_PORTRAIT = 0.477f
    private const val TAN_HALF_FOV_X_LANDSCAPE = 0.600f

    val YOLO_CLASS_ALIASES = mapOf(
        "staircase" to "stairs",
        "flight of stairs" to "stairs",
        "steps" to "stairs",
        "stair" to "stairs",
        "doorstep" to "step",
        "curb" to "step",
        "bump" to "step",
        "water puddle" to "pothole",
        "puddle" to "pothole",
        "open manhole" to "open_manhole",
        "manhole" to "open_manhole",
        "glass door" to "glass_door",
        "pillar" to "column",
        "pole" to "column",
        "bollard" to "column",
        "barrier" to "obstacle",
        "trash can" to "obstacle",
        "waste_container" to "obstacle",
        "fence" to "obstacle",
        "hole" to "dropoff",
        "animal" to "dog",
    )

    val VOCAB_VN = mapOf(
        "person" to "người",
        "bicycle" to "xe đạp",
        "car" to "xe ô tô",
        "motorcycle" to "xe máy",
        "bus" to "xe buýt",
        "truck" to "xe tải",
        "bench" to "ghế dài",
        "cat" to "mèo",
        "dog" to "chó",
        "backpack" to "ba lô",
        "suitcase" to "vali",
        "bottle" to "chai nước",
        "cup" to "cốc nước",
        "chair" to "ghế",
        "couch" to "ghế sofa",
        "potted plant" to "chậu cây",
        "bed" to "giường ngủ",
        "dining table" to "bàn ăn",
        "table" to "bàn",
        "tv" to "màn hình tivi",
        "laptop" to "máy tính xách tay",
        "book" to "cuốn sách",
        "refrigerator" to "tủ lạnh",
        "door" to "cửa ra vào",
        "stair" to "bậc thang",
        "stairs" to "cầu thang",
        "wall" to "bức tường",
        "column" to "cột nhà",
        "pillar" to "cột nhà",
        "cabinet" to "tủ",
        "step" to "bậc tam cấp",
        "curb" to "bậc tam cấp",
        "doorstep" to "bậc thềm",
        "threshold" to "bậc thềm",
        "multi_obstacle" to "nhiều vật cản",
        "obstacle" to "vật cản",
        "mirror" to "tấm gương",
        "glass_door" to "cửa kính",
        "open_manhole" to "hố ga hở",
        "pothole" to "ổ gà",
        "dropoff" to "hố sâu",
    )

    private val OBJECT_REAL_HEIGHTS_M = mapOf(
        "bottle" to 0.24f,
        "wine glass" to 0.18f,
        "cup" to 0.12f,
        "bowl" to 0.10f,
        "cell phone" to 0.15f,
        "book" to 0.22f,
        "laptop" to 0.25f,
        "potted plant" to 0.45f,
        "backpack" to 0.45f,
        "suitcase" to 0.65f,
        "chair" to 0.85f,
        "couch" to 0.85f,
        "table" to 0.75f,
        "dining table" to 0.75f,
        "desk" to 0.75f,
        "bed" to 0.70f,
        "bench" to 0.80f,
        "tv" to 0.55f,
        "refrigerator" to 1.65f,
        "door" to 2.00f,
        "cabinet" to 1.50f,
        "car" to 1.50f,
        "motorcycle" to 1.10f,
        "bicycle" to 1.00f,
        "bus" to 3.00f,
        "truck" to 2.80f,
        "dog" to 0.50f,
        "cat" to 0.30f,
    )

    private val TABLETOP_CLASSES = setOf(
        "bottle", "wine glass", "cup", "bowl", "cell phone", "book", "laptop"
    )

    val GROUND_DEFECT_CLASSES = setOf("pothole", "open_manhole", "dropoff", "step")

    val POTHOLE_GROUND_LABELS = setOf(
        "earth", "road", "route", "sidewalk", "pavement", "path", "ground", "water", "sand", "grass", "field"
    )

    val POTHOLE_NON_GROUND_LABELS = setOf(
        "stairway", "stairs", "step", "escalator", "grandstand", "tree", "plant", "wall", "building",
        "house", "sky", "person", "sofa", "chair", "armchair", "table", "coffee table", "desk",
        "cabinet", "door", "fence", "railing", "bannister", "mirror"
    )

    val SEG_OBSTACLE_MAP = mapOf(
        "wall" to ("wall" to "bức tường"),
        "door" to ("door" to "cửa ra vào"),
        "double door" to ("door" to "cửa ra vào"),
        "screen door" to ("door" to "cửa ra vào"),
        "stairway" to ("stairs" to "cầu thang"),
        "stairs" to ("stairs" to "cầu thang"),
        "step" to ("stairs" to "bậc thang"),
        "escalator" to ("stairs" to "thang cuốn"),
        "grandstand" to ("stairs" to "cầu thang"),
        "mirror" to ("mirror" to "tấm gương"),
        "column" to ("column" to "cột nhà"),
        "pillar" to ("column" to "cột nhà"),
        "cabinet" to ("cabinet" to "tủ"),
        "wardrobe" to ("cabinet" to "tủ quần áo"),
        "bookcase" to ("cabinet" to "tủ sách"),
        "chest of drawers" to ("cabinet" to "tủ ngăn kéo"),
        "shelf" to ("cabinet" to "kệ tủ"),
        "refrigerator" to ("refrigerator" to "tủ lạnh"),
        "fence" to ("obstacle" to "hàng rào"),
        "railing" to ("obstacle" to "lan can"),
    )

    fun normalizeClass(rawName: String): String {
        val key = rawName.lowercase().trim()
        return YOLO_CLASS_ALIASES[key] ?: key
    }

    fun getVietnameseName(className: String): String {
        val key = className.lowercase().trim()
        return VOCAB_VN[key] ?: key.ifEmpty { "vật cản" }
    }

    fun determinePosition(x1: Int, x2: Int, imgW: Int): String {
        val cx = (x1 + x2) * 0.5f
        val third = imgW.coerceAtLeast(1) / 3.0f
        return when {
            cx < third -> "left"
            cx > 2f * third -> "right"
            else -> "center"
        }
    }

    fun isInRoi(box: EdgeBox, imgW: Int, imgH: Int): Boolean {
        val objBottomX = (box.x1 + box.x2) * 0.5f
        val objBottomY = box.y2.toFloat()
        val topY = imgH * 0.40f
        if (objBottomY < topY) return false
        val centerX = imgW * 0.5f
        val topWidth = imgW * 0.40f
        val botWidth = imgW * 0.80f
        val progress = ((objBottomY - topY) / max(1f, imgH - topY)).coerceIn(0f, 1f)
        val curHalfW = (topWidth * 0.5f) + progress * ((botWidth - topWidth) * 0.5f)
        return abs(objBottomX - centerX) <= curHalfW
    }

    fun round1(v: Float): Float = (v * 10f).roundToInt() / 10f

    fun round2(v: Float): Float = (v * 100f).roundToInt() / 100f

    fun estimatePinholeDistanceM(className: String, box: EdgeBox, imgW: Int, imgH: Int): Float {
        val w = max(1, imgW).toFloat()
        val h = max(1, imgH).toFloat()
        val boxH = max(1f, (box.y2 - box.y1).toFloat())
        val boxW = max(1f, (box.x2 - box.x1).toFloat())

        val isPortrait = h > w
        val tanY = if (isPortrait) TAN_HALF_FOV_Y_PORTRAIT else TAN_HALF_FOV_Y_LANDSCAPE
        val tanX = if (isPortrait) TAN_HALF_FOV_X_PORTRAIT else TAN_HALF_FOV_X_LANDSCAPE
        val fy = (h * 0.5f) / tanY
        val fx = (w * 0.5f) / tanX

        val touchesBottom = box.y2 >= h * 0.95f
        val touchesTop = box.y1 <= h * 0.05f
        val cLower = className.lowercase().trim()

        var dist: Float
        if (cLower == "person") {
            val dH = fy * PERSON_HEIGHT_M / boxH
            val dW = fx * PERSON_SHOULDER_WIDTH_M / boxW
            dist = if (touchesBottom || touchesTop) dW else dH
        } else if (cLower in TABLETOP_CLASSES) {
            val realH = OBJECT_REAL_HEIGHTS_M[cLower] ?: 0.22f
            dist = fy * realH / boxH
        } else {
            val horizonY = h * 0.35f
            val below = box.y2.toFloat() - horizonY
            val dGround = if (below <= 1f) {
                max(1.2f, 2.5f * (h / max(boxH, 1f)) * 0.30f)
            } else {
                fy * CAMERA_HEIGHT_M / below
            }
            val realObjH = OBJECT_REAL_HEIGHTS_M[cLower]
            if (realObjH != null) {
                val dObj = fy * realObjH / boxH
                dist = if (!touchesTop && !touchesBottom) {
                    0.70f * dObj + 0.30f * dGround
                } else {
                    min(dGround, dObj)
                }
            } else {
                dist = dGround
            }
            if ((boxH / h) >= 0.65f && touchesBottom) {
                dist = min(dist, 1.1f)
            }
        }
        return round1(dist.coerceIn(0.3f, MAX_DISTANCE_M))
    }

    fun resolveCalibratedDistance(
        className: String,
        metricDist: Float?,
        pinholeDist: Float,
        box: EdgeBox,
        imgW: Int,
        imgH: Int,
    ): Float {
        val w = max(1, imgW).toFloat()
        val h = max(1, imgH).toFloat()
        val bw = max(1f, (box.x2 - box.x1).toFloat())
        val bh = max(1f, (box.y2 - box.y1).toFloat())
        val boxHRatio = bh / h
        val areaRatio = (bw * bh) / (w * h)

        if (metricDist == null || metricDist < 0.2f || metricDist > MAX_DISTANCE_M) {
            return round1(pinholeDist)
        }

        val touchesBottom = box.y2 >= h * 0.94f
        val touchesTop = box.y1 <= h * 0.05f
        val cLower = className.lowercase().trim()
        val hasKnownHeight = (cLower == "person") || (cLower in OBJECT_REAL_HEIGHTS_M)

        // 0. Vật thể trùm gần kín khung hình (bàn, tường, tủ sát camera) -> không bao giờ đoán xa > 1.4m
        if (areaRatio >= 0.45f || boxHRatio >= 0.65f) {
            val safeClose = min(metricDist, max(0.4f, min(1.4f, pinholeDist * 1.3f)))
            return round1(safeClose)
        }

        // 0b. Vật sát mặt (<= 1.35m theo Pinhole hoặc hộp lớn) nhưng DepthV2 bị chói đèn/nền sau kéo xa > 2.0m
        if ((pinholeDist <= 1.35f || areaRatio >= 0.28f || boxHRatio >= 0.48f) && metricDist > max(2.0f, pinholeDist * 1.75f)) {
            val blendedClose = 0.75f * pinholeDist + 0.25f * min(metricDist, pinholeDist * 1.6f)
            return round1(blendedClose.coerceIn(0.3f, 1.6f))
        }

        if (pinholeDist >= 3.0f && boxHRatio < 0.22f && metricDist < pinholeDist * 0.62f) {
            return round1(pinholeDist)
        }
        if (hasKnownHeight && !touchesBottom && !touchesTop && boxHRatio < 0.36f &&
            pinholeDist >= 2.35f && metricDist < pinholeDist * 0.75f
        ) {
            return round1(0.25f * metricDist + 0.75f * pinholeDist)
        }
        return round1(metricDist)
    }

    fun medianOfValues(values: FloatArray, count: Int): Float? {
        if (count <= 0) return null
        val slice = values.copyOf(count)
        slice.sort()
        val mid = count / 2
        return if (count % 2 == 1) slice[mid] else (slice[mid - 1] + slice[mid]) * 0.5f
    }

    fun getBoxMedianDepth(
        box: EdgeBox,
        dmap: FloatArray?,
        dW: Int,
        dH: Int,
        imgW: Int,
        imgH: Int,
    ): Float? {
        if (dmap == null || dmap.size < dW * dH || dW <= 0 || dH <= 0) return null
        val sx1 = ((box.x1 + 0.22f * box.width) * dW / max(1, imgW)).toInt().coerceIn(0, dW - 1)
        val sy1 = ((box.y1 + 0.18f * box.height) * dH / max(1, imgH)).toInt().coerceIn(0, dH - 1)
        val sx2 = ((box.x2 - 0.22f * box.width) * dW / max(1, imgW)).toInt().coerceIn(sx1 + 1, dW)
        val sy2 = ((box.y2 - 0.32f * box.height) * dH / max(1, imgH)).toInt().coerceIn(sy1 + 1, dH)

        val stepX = max(1, (sx2 - sx1) / 18)
        val stepY = max(1, (sy2 - sy1) / 18)
        val buf = FloatArray(((sx2 - sx1) / stepX + 2) * ((sy2 - sy1) / stepY + 2))
        var n = 0
        var y = sy1
        while (y < sy2) {
            val rowOff = y * dW
            var x = sx1
            while (x < sx2) {
                if (n < buf.size) {
                    buf[n++] = dmap[rowOff + x]
                }
                x += stepX
            }
            y += stepY
        }
        val med = medianOfValues(buf, n) ?: return null
        return round1(med.coerceIn(0.2f, MAX_DISTANCE_M))
    }

    fun estimateSurfaceRingDepth(
        box: EdgeBox,
        dmap: FloatArray?,
        dW: Int,
        dH: Int,
        imgW: Int,
        imgH: Int,
    ): Float {
        val bhRatio = box.height / max(1f, imgH.toFloat())
        val fallbackD = max(0.8f, round1(1.3f / max(0.30f, bhRatio)))
        if (dmap == null || dmap.size < dW * dH) return fallbackD.coerceAtMost(3.5f)

        val sx1 = (box.x1 * dW / max(1, imgW)).toInt().coerceIn(0, dW - 1)
        val sy1 = (box.y1 * dH / max(1, imgH)).toInt().coerceIn(0, dH - 1)
        val sx2 = (box.x2 * dW / max(1, imgW)).toInt().coerceIn(sx1 + 1, dW)
        val sy2 = (box.y2 * dH / max(1, imgH)).toInt().coerceIn(sy1 + 1, dH)
        val pad = max(4, (0.06f * min(dW, dH)).toInt())

        val ringSamples = FloatArray(dW * 4 + dH * 4)
        var n = 0
        val leftX1 = max(0, sx1 - pad)
        val rightX2 = min(dW, sx2 + pad)
        for (y in sy1 until sy2 step 2) {
            val off = y * dW
            for (x in leftX1 until sx1) if (n < ringSamples.size) ringSamples[n++] = dmap[off + x]
            for (x in sx2 until rightX2) if (n < ringSamples.size) ringSamples[n++] = dmap[off + x]
        }
        val botY1 = min(dH - 1, sy2)
        val botY2 = min(dH, sy2 + pad)
        for (y in botY1 until botY2) {
            val off = y * dW
            for (x in sx1 until sx2 step 2) if (n < ringSamples.size) ringSamples[n++] = dmap[off + x]
        }
        val ringMed = medianOfValues(ringSamples, n) ?: return fallbackD.coerceAtMost(3.5f)
        val innerMed = getBoxMedianDepth(box, dmap, dW, dH, imgW, imgH) ?: ringMed
        val est = if (innerMed > ringMed * 1.45f) ringMed else min(ringMed, innerMed)
        return round1(est.coerceIn(0.4f, min(fallbackD + 0.5f, 4.0f)))
    }

    fun estimatePotholeDistance(
        box: EdgeBox,
        dmap: FloatArray?,
        dW: Int,
        dH: Int,
        imgW: Int,
        imgH: Int,
    ): Float {
        if (dmap != null && dmap.size >= dW * dH) {
            val sx1 = ((box.x1 + 0.20f * box.width) * dW / max(1, imgW)).toInt().coerceIn(0, dW - 1)
            val sy1 = ((box.y1 + 0.65f * box.height) * dH / max(1, imgH)).toInt().coerceIn(0, dH - 1)
            val sx2 = ((box.x2 - 0.20f * box.width) * dW / max(1, imgW)).toInt().coerceIn(sx1 + 1, dW)
            val sy2 = ((box.y1 + 0.95f * box.height) * dH / max(1, imgH)).toInt().coerceIn(sy1 + 1, dH)
            val vals = FloatArray(((sx2 - sx1) + 1) * ((sy2 - sy1) + 1))
            var n = 0
            for (y in sy1 until sy2) {
                val off = y * dW
                for (x in sx1 until sx2) {
                    vals[n++] = dmap[off + x]
                }
            }
            if (n > 0) {
                vals.sort(0, n)
                val q20 = vals[(n * 0.20f).toInt().coerceIn(0, n - 1)]
                return round1(q20.coerceIn(0.5f, 6.0f))
            }
        }
        val dist = max(0.8f, 2.8f - 1.8f * (box.y2 / max(1f, imgH.toFloat())))
        return round1(dist)
    }

    fun analyzeThreeLanesDepth(dmap: FloatArray?, dW: Int, dH: Int): ThreeLaneDepth {
        if (dmap == null || dmap.size < dW * dH || dW <= 0 || dH <= 0) {
            return ThreeLaneDepth(hasDepth = false, lanes = emptyMap())
        }
        val y1 = (dH * 0.22f).toInt()
        val y2 = (dH * 0.82f).toInt().coerceAtLeast(y1 + 1)
        val xLeft = (dW * 0.33f).toInt().coerceAtLeast(1)
        val xRight = (dW * 0.67f).toInt().coerceIn(xLeft + 1, dW)

        fun calcSlice(sx1: Int, sx2: Int): LaneDepthState {
            val step = 3
            val buf = FloatArray(((y2 - y1) / step + 2) * ((sx2 - sx1) / step + 2))
            var n = 0
            var closeCnt = 0
            var midCnt = 0
            var y = y1
            while (y < y2) {
                val off = y * dW
                var x = sx1
                while (x < sx2) {
                    val v = dmap[off + x]
                    if (n < buf.size) buf[n++] = v
                    if (v < 1.00f) closeCnt++
                    if (v < 2.00f) midCnt++
                    x += step
                }
                y += step
            }
            if (n == 0) return LaneDepthState(4.0f, 0f, 0f, false)
            val med = round2(medianOfValues(buf, n) ?: 3.5f)
            val closeR = round2(closeCnt.toFloat() / n.toFloat())
            val midR = round2(midCnt.toFloat() / n.toFloat())
            return LaneDepthState(
                medianM = med,
                closeRatio = closeR,
                midRatio = midR,
                isBlocked = (closeR >= 0.42f || med <= 1.00f),
            )
        }

        return ThreeLaneDepth(
            hasDepth = true,
            lanes = mapOf(
                "left" to calcSlice(0, xLeft),
                "center" to calcSlice(xLeft, xRight),
                "right" to calcSlice(xRight, dW),
            ),
        )
    }
}
