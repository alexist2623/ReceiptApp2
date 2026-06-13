package com.receiptapp.util

inline fun <T> runCatchingMessage(block: () -> T): Pair<T?, String?> {
    return try {
        block() to null
    } catch (throwable: Throwable) {
        null to (throwable.message ?: throwable::class.java.simpleName)
    }
}
