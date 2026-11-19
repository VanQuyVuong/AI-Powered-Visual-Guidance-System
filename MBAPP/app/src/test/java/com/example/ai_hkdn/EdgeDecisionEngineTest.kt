package com.example.ai_hkdn

import com.example.ai_hkdn.edge.DistanceEstimator
import com.example.ai_hkdn.edge.EdgeBox
import com.example.ai_hkdn.edge.EdgeCandidate
import com.example.ai_hkdn.edge.EdgeDecisionEngine
import com.example.ai_hkdn.edge.LaneDepthState
import com.example.ai_hkdn.edge.ThreeLaneDepth
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class EdgeDecisionEngineTest {

    @Test
    fun distanceEstimator_resolvesFarChairAndRingMirrorCorrectly() {
        val farChairBox = EdgeBox(x1 = 200, y1 = 190, x2 = 280, y2 = 300)
        val pinhole = DistanceEstimator.estimatePinholeDistanceM("chair", farChairBox, 480, 640)
        assertTrue("Chair small box should estimate >= 2.35m, got $pinhole", pinhole >= 2.35f)

        val resolved = DistanceEstimator.resolveCalibratedDistance("chair", 1.1f, pinhole, farChairBox, 480, 640)
        assertTrue("Calibrated far chair must not false alarm close, got $resolved", resolved >= 2.0f)

        assertEquals("tấm gương", DistanceEstimator.getVietnameseName("mirror"))
        assertEquals("cửa kính", DistanceEstimator.getVietnameseName("glass_door"))
        assertEquals("stairs", DistanceEstimator.normalizeClass("flight of stairs"))
    }

    @Test
    fun decisionEngine_triggersStairsAndFiltersReflectionsInsideMirror() {
        val engine = EdgeDecisionEngine()
        val mirrorBox = EdgeBox(60, 20, 420, 440)
        val reflectionPersonBox = EdgeBox(150, 120, 320, 400)

        val lanes = ThreeLaneDepth(
            hasDepth = true,
            lanes = mapOf(
                "left" to LaneDepthState(2.5f, 0f, 0.2f, false),
                "center" to LaneDepthState(1.6f, 0.1f, 0.7f, false),
                "right" to LaneDepthState(2.5f, 0f, 0.2f, false),
            )
        )

        val outMirror = engine.processFrame(
            rawCandidates = listOf(
                EdgeCandidate("mirror", "tấm gương", 0.9f, "center", 1.6f, true, true, mirrorBox, "YOLOE"),
                EdgeCandidate("person", "người", 0.85f, "center", 1.6f, true, true, reflectionPersonBox, "YOLOE"),
            ),
            depthLanes = lanes,
            imgW = 480,
            imgH = 640,
            yoloeCount = 2,
            auxHazardCount = 0,
            segHazardCount = 0,
            runtimeLabel = "NNAPI-FP16",
            nowMs = 10_000L,
        )
        assertEquals(1, outMirror.detectedObjects.size)
        assertEquals("mirror", outMirror.detectedObjects[0].className)
        assertTrue(outMirror.displayText.contains("tấm gương"))

        engine.reset()
        val outStairs = engine.processFrame(
            rawCandidates = listOf(
                EdgeCandidate("stairs", "cầu thang", 0.92f, "center", 1.5f, true, true, EdgeBox(40, 300, 440, 620), "YOLOE")
            ),
            depthLanes = lanes,
            imgW = 480,
            imgH = 640,
            yoloeCount = 1,
            auxHazardCount = 0,
            segHazardCount = 0,
            runtimeLabel = "NNAPI-FP16",
            nowMs = 20_000L,
        )
        assertEquals("STEPS", outStairs.stateCode)
        assertTrue(outStairs.guidance.text.contains("cầu thang"))
        assertFalse(outStairs.guidance.text.isBlank())
    }
}
