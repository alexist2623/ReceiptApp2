package com.receiptapp.inference.relg

import java.io.File
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

class SpanRelGSchema private constructor(
    val field2Id: Map<String, Int>,
    val kind2Id: Map<String, Int>,
) {
    fun fieldId(field: String): Int? = field2Id[canonicalField(field)]
    fun kindId(kind: String): Int = kind2Id[kind] ?: error("Missing rel-g kind id: $kind")
    fun isHead(field: String): Boolean = canonicalField(field) in headFields
    fun isDependent(field: String): Boolean = canonicalField(field) in dependentFields
    fun isCandidateDependent(field: String): Boolean = canonicalField(field) in candidateDependentFields

    companion object {
        private val json = Json { ignoreUnknownKeys = true }

        val headFields = setOf(
            "ITEM_NAME",
            "SUBTOTAL_NAME",
            "TAX_NAME",
            "TOTAL_NAME",
            "TIP_NAME",
        )

        val itemDependentFields = setOf(
            "ITEM_PRICE",
            "ITEM_QTY",
            "ITEM_UNIT_PRICE",
            "ITEM_CODE",
            "ITEM_SKU",
            "ITEM_DISCOUNT",
            "ITEM_OPTION",
            "ITEM_TAX_FLAG",
            "ITEM_ETC",
        )

        val summaryDependentFields = setOf(
            "SUBTOTAL_PRICE",
            "TAX_PRICE",
            "TAX_RATE",
            "TOTAL_PRICE",
            "TIP_PRICE",
        )

        val dependentFields = itemDependentFields + summaryDependentFields

        val hardNegativeFields = setOf(
            "STORE_NAME",
            "STORE_ADDRESS",
            "STORE_PHONE",
            "DATE",
            "TIME",
            "RECEIPT_ID",
            "SUBTOTAL_NAME",
            "SUBTOTAL_PRICE",
            "TAX_NAME",
            "TAX_RATE",
            "TAX_PRICE",
            "DISCOUNT_NAME",
            "DISCOUNT_PRICE",
            "SERVICE_NAME",
            "SERVICE_PRICE",
            "TOTAL_NAME",
            "TOTAL_PRICE",
            "CASH_NAME",
            "CASH_PRICE",
            "CHANGE_NAME",
            "CHANGE_PRICE",
            "CARD_NAME",
            "CARD_PRICE",
            "TIP_NAME",
            "TIP_PRICE",
            "PAYMENT_METHOD",
            "PAYMENT_CARD",
            "PAYMENT_AUTH_CODE",
            "PAYMENT_INFO",
            "APPROVAL_CODE",
            "TRANSACTION_ID",
        )

        val candidateDependentFields = dependentFields + hardNegativeFields

        fun fromFile(file: File): SpanRelGSchema {
            val root = json.parseToJsonElement(file.readText()).jsonObject
            val field2Id = parseStringIntMap(root["field2id"]?.jsonObject)
            val kind2Id = parseStringIntMap(root["kind2id"]?.jsonObject)
            require(field2Id.isNotEmpty()) { "span rel-g schema missing field2id: ${file.absolutePath}" }
            require(kind2Id.isNotEmpty()) { "span rel-g schema missing kind2id: ${file.absolutePath}" }
            return SpanRelGSchema(field2Id, kind2Id)
        }

        fun canonicalField(value: String?): String {
            var field = value?.trim().orEmpty()
            if (field.isEmpty() || field == "O") return "O"
            if (field.startsWith("B-") || field.startsWith("I-")) field = field.substring(2)
            field = field.uppercase()
                .replace(".", "_")
                .replace("-", "_")
                .replace("/", "_")
                .replace(" ", "_")
            return when (field) {
                "MENU_NM", "MENU_NAME" -> "ITEM_NAME"
                "MENU_PRICE" -> "ITEM_PRICE"
                "MENU_CNT", "ITEM_CNT" -> "ITEM_QTY"
                "MENU_UNITPRICE", "MENU_UNIT_PRICE" -> "ITEM_UNIT_PRICE"
                "MENU_NUM", "MENU_CODE" -> "ITEM_CODE"
                "MENU_SUB_NM", "MENU_OPTION", "ITEM_SUB_NAME" -> "ITEM_OPTION"
                "MENU_DISCOUNTPRICE", "MENU_DISCOUNT_PRICE" -> "ITEM_DISCOUNT"
                "SUBTOTAL_SUBTOTAL_PRICE" -> "SUBTOTAL_PRICE"
                "SUBTOTAL_TAX_PRICE" -> "TAX_PRICE"
                "TOTAL_TOTAL_PRICE" -> "TOTAL_PRICE"
                "TOTAL_CASHPRICE" -> "CASH_PRICE"
                "TOTAL_CHANGEPRICE" -> "CHANGE_PRICE"
                "TOTAL_CREDITCARDPRICE" -> "CARD_PRICE"
                else -> field
            }
        }

        private fun parseStringIntMap(obj: JsonObject?): Map<String, Int> {
            if (obj == null) return emptyMap()
            return obj.mapNotNull { (key, value) ->
                val id = value.jsonPrimitive.intOrNull
                    ?: value.jsonPrimitive.contentOrNull?.toIntOrNull()
                id?.let { key to it }
            }.toMap()
        }
    }
}
