package com.receiptapp.network

import org.junit.Assert.assertEquals
import org.junit.Test

class ServerConfigTest {
    @Test
    fun normalizesServerUrl() {
        assertEquals("", ServerConfig.normalizeBaseUrl("  "))
        assertEquals("http://192.168.0.10:8000", ServerConfig.normalizeBaseUrl("192.168.0.10:8000/"))
        assertEquals("https://example.com/api", ServerConfig.normalizeBaseUrl("https://example.com/api/"))
    }
}
