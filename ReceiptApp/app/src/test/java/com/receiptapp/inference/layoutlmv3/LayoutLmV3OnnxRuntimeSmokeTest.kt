package com.receiptapp.inference.layoutlmv3

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeNoException
import org.junit.Assume.assumeNotNull
import org.junit.Test

class LayoutLmV3OnnxRuntimeSmokeTest {
    @Test
    fun runsCordInt8OnnxWithPrecomputedFixtureTensors() {
        val located = LayoutLmV3ParityFixture.locate()
        assumeNotNull("LayoutLMv3 model/fixture artifacts are not present; skipping ONNX runtime smoke.", located)
        val fixture = located!!

        val inputs = fixture.loadInputs()
        val outputs = try {
            LayoutLmV3OnnxRuntimeSmokeRunner.run(fixture.modelPath, inputs)
        } catch (error: UnsatisfiedLinkError) {
            assumeNoException("ONNX Runtime JVM native library is unavailable on this host.", error)
            return
        }
        val expected = fixture.loadExpectedWordLabels()
        val agreement = LayoutLmV3OnnxRuntimeSmokeRunner.compareExpectedWordLabels(outputs.logits, expected)

        assertEquals(listOf(1, 512, 59), outputs.logitsShape)
        assertEquals(1, outputs.lastHiddenStateShape[0])
        assertEquals(768, outputs.lastHiddenStateShape[2])
        assertTrue(
            "Expected word label agreement must be 1.0; mismatches=${agreement.mismatches.take(5)}",
            agreement.checked > 0 && agreement.agreement >= 1.0,
        )
    }
}
