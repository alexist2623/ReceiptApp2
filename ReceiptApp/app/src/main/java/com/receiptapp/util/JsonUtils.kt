package com.receiptapp.util

import kotlinx.serialization.encodeToString
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.json.Json

@OptIn(ExperimentalSerializationApi::class)
object JsonUtils {
    val prettyJson: Json = Json {
        prettyPrint = true
        explicitNulls = false
        encodeDefaults = true
        ignoreUnknownKeys = true
    }

    inline fun <reified T> encodePretty(value: T): String = prettyJson.encodeToString(value)

    inline fun <reified T> decode(value: String): T = prettyJson.decodeFromString(value)
}
