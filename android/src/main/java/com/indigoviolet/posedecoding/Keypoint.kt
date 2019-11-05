package com.indigoviolet.posedecoding

data class Coords<T> (val y: T, val x: T)
typealias IntCoords = Coords<Int>
typealias FloatCoords = Coords<Float>

data class Keypoint(val part: String, val partId: Int, val coords: FloatCoords, val score: Float) {
    fun toMap(): Map<String, Any> {
        val pos: Map<String, Float> = mapOf("y" to coords.y, "x" to coords.x)
        return mapOf("score" to score, "partId" to partId, "part" to part, "position" to pos)
    }

    val y get() = coords.y
    val x get() = coords.x
}
