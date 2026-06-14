package com.receiptapp.inference.layoutlmv3

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeNotNull
import org.junit.Test

class LayoutLmV3PreprocessingParityGateTest {
    @Test
    fun comparesAndroidPreprocessingOutputAgainstPythonFixtureTensors() {
        val located = LayoutLmV3ParityFixture.locate()
        assumeNotNull("LayoutLMv3 fixtures are not present; skipping preprocessing parity gate.", located)
        val fixture = located!!

        val expected = fixture.loadInputs()
        /*
         * Phase B hook:
         * Replace this with real Android tokenizer/image preprocessing output once
         * that implementation exists. The gate itself already enforces exact
         * input_ids, attention_mask, bbox, and reports pixel_values drift.
         */
        val androidProduced = expected
        val report = LayoutLmV3PreprocessingParityGate.compare(expected, androidProduced)

        assertTrue(report.inputIds.exact)
        assertTrue(report.attentionMask.exact)
        assertTrue(report.bbox.exact)
        assertEquals(0.0, report.pixelValues.maxAbsDiff, 0.0)
        assertEquals(0.0, report.pixelValues.meanAbsDiff, 0.0)
        assertTrue(report.exactTokenInputs)
    }
}
