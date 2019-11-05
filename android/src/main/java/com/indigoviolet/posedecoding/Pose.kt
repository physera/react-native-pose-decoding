package com.indigoviolet.posedecoding

data class Pose(val score: Float, val keypoints: List<Keypoint>) {
    fun toMap(): Map<String, Any> {
        val kpMap: List<Map<String, Any>> = keypoints.map { it.toMap() }
        return mapOf("score" to score, "keypoints" to kpMap)
    }
}
