package com.indigoviolet.posedecoding

import java.util.*
import kotlin.math.exp
import kotlin.math.roundToInt

typealias ThreeDFloatArray = Array<Array<FloatArray>>
typealias FourDFloatArray = Array<ThreeDFloatArray>

class PosenetDecoder private constructor(
    private val model: TfliteModel,
    private val params: DecoderParams
) : Decoder(model, params) {

    companion object {
        private lateinit var instance: PosenetDecoder

        @JvmStatic
        fun getInstance(model: TfliteModel, params: DecoderParams): PosenetDecoder {
            // This is only to try to keep the same instance shared
            // across multiple invocations of decode, to avoid
            // reinitialization. It's not important that this be
            // synchronized -- there is no actual shared state across
            // threads
            val existsAndMatches = ::instance.isInitialized && (
                instance.model == model &&
                instance.params == params
            )
            if (!existsAndMatches) {
                instance = PosenetDecoder(model, params)
            }

            return instance
        }
    }

    private val partNames = arrayOf("nose",
            "leftEye", "rightEye",
            "leftEar", "rightEar",
            "leftShoulder", "rightShoulder",
            "leftElbow", "rightElbow",
            "leftWrist", "rightWrist",
            "leftHip", "rightHip",
            "leftKnee", "rightKnee",
            "leftAnkle", "rightAnkle")

    private val poseChain = arrayOf(
            Pair("nose", "leftEye"),
            Pair("leftEye", "leftEar"),
            Pair("nose", "rightEye"),
            Pair("rightEye", "rightEar"),
            Pair("nose", "leftShoulder"),
            Pair("leftShoulder", "leftElbow"),
            Pair("leftElbow", "leftWrist"),
            Pair("leftShoulder", "leftHip"),
            Pair("leftHip", "leftKnee"),
            Pair("leftKnee", "leftAnkle"),
            Pair("nose", "rightShoulder"),
            Pair("rightShoulder", "rightElbow"),
            Pair("rightElbow", "rightWrist"),
            Pair("rightShoulder", "rightHip"),
            Pair("rightHip", "rightKnee"),
            Pair("rightKnee", "rightAnkle")
    )

    private val partsIds: Map<String, Int>
    private val parentToChildEdges: List<Int>
    private val childToParentEdges: List<Int>

    init {
        partsIds = partNames.indices.map { Pair(partNames[it], it) }.toMap()
        parentToChildEdges = poseChain.map { partsIds.getValue(it.second) }
        childToParentEdges = poseChain.map { partsIds.getValue(it.first) }
    }

    override fun addPixelValue(v: Int) {
        inputBuf.putFloat((((v shr 16) and 0xFF) - params.posenetMean) / params.posenetStd)
        inputBuf.putFloat((((v shr 8) and 0xFF) - params.posenetMean) / params.posenetStd)
        inputBuf.putFloat(((v and 0xFF) - params.posenetMean) / params.posenetStd)
    }

    override fun decode(): List<Pose> {
        val localMaximumRadius = 1

        @Suppress("UNCHECKED_CAST")
        val scores = (outputBuf[0] as FourDFloatArray)[0]

        @Suppress("UNCHECKED_CAST")
        val offsets = (outputBuf[1] as FourDFloatArray)[0]

        @Suppress("UNCHECKED_CAST")
        val displacementsFwd = (outputBuf[2] as FourDFloatArray)[0]

        @Suppress("UNCHECKED_CAST")
        val displacementsBwd = (outputBuf[3] as FourDFloatArray)[0]

        val pq = buildPartWithScoreQueue(scores, params.threshold.toDouble(), localMaximumRadius)

        val squaredNmsRadius = params.posenetNMSRadius * params.posenetNMSRadius

        val poses = mutableListOf<Pose>()
        while (poses.size < params.posenetNumResults && pq.size > 0) {
            val root = pq.poll()
            val rootPoint = getImageCoords(root, offsets)

            if (withinNmsRadiusOfCorrespondingPoint(poses, squaredNmsRadius.toFloat(), rootPoint, root.partId))
                continue

            val keypoints = decodePose(root, rootPoint, scores, offsets, displacementsFwd, displacementsBwd)
            val score = getInstanceScore(keypoints, poses, squaredNmsRadius)
            poses.add(Pose(score, keypoints))
        }
        return poses
    }

    private fun decodePose(root: Keypoint, rootPoint: FloatCoords, scores: ThreeDFloatArray, offsets: ThreeDFloatArray, displacementsFwd: ThreeDFloatArray, displacementsBwd: ThreeDFloatArray): List<Keypoint> {
        val rootKeypoint = Keypoint(partNames[root.partId], root.partId, rootPoint, root.score)

        val keypoints = mutableMapOf<Int, Keypoint>()
        keypoints[root.partId] = rootKeypoint

        for (edge in parentToChildEdges.indices.reversed()) {
            val sourceKeypointId = parentToChildEdges[edge]
            val targetKeypointId = childToParentEdges[edge]
            if (keypoints.containsKey(sourceKeypointId) && !keypoints.containsKey(targetKeypointId)) {
                val keypoint = traverseToTargetKeypoint(edge, keypoints[sourceKeypointId]!!, targetKeypointId, scores, offsets, displacementsBwd)
                keypoints[targetKeypointId] = keypoint
            }
        }

        for (edge in parentToChildEdges.indices) {
            val sourceKeypointId = childToParentEdges[edge]
            val targetKeypointId = parentToChildEdges[edge]
            if (keypoints.containsKey(sourceKeypointId) && !keypoints.containsKey(targetKeypointId)) {
                val keypoint = traverseToTargetKeypoint(edge, keypoints[sourceKeypointId]!!, targetKeypointId, scores, offsets, displacementsFwd)
                keypoints[targetKeypointId] = keypoint
            }
        }
        return keypoints.values.toList()
    }

    private fun buildPartWithScoreQueue(scores: ThreeDFloatArray, threshold: Double, localMaximumRadius: Int): PriorityQueue<Keypoint> {
        val pq = PriorityQueue(1, compareByDescending<Keypoint> { it.score })

        for (heatmapY in scores.indices) {
            for (heatmapX in scores[0].indices) {
                for (keypointId in scores[0][0].indices) {
                    val score = sigmoid(scores[heatmapY][heatmapX][keypointId])
                    if (score < threshold) continue

                    if (scoreIsMaximumInLocalWindow(keypointId, score, IntCoords(heatmapY, heatmapX), localMaximumRadius, scores)) {
                        val res = Keypoint(partNames[keypointId], keypointId, FloatCoords(heatmapY.toFloat(), heatmapX.toFloat()), score)
                        pq.add(res)
                    }
                }
            }
        }

        return pq
    }

    private fun scoreIsMaximumInLocalWindow(keypointId: Int,
                                            score: Float,
                                            heatmapIdxs: IntCoords,
                                            localMaximumRadius: Int,
                                            scores: ThreeDFloatArray): Boolean {
        var localMaximum = true
        val height = scores.size
        val width = scores[0].size

        val yStart = maxOf(heatmapIdxs.y - localMaximumRadius, 0)
        val yEnd = minOf(heatmapIdxs.y + localMaximumRadius + 1, height)
        for (yCurrent in yStart until yEnd) {
            val xStart = maxOf(heatmapIdxs.x - localMaximumRadius, 0)
            val xEnd = minOf(heatmapIdxs.x + localMaximumRadius + 1, width)
            for (xCurrent in xStart until xEnd) {
                if (sigmoid(scores[yCurrent][xCurrent][keypointId]) > score) {
                    localMaximum = false
                    break
                }
            }
            if (!localMaximum) {
                break
            }
        }

        return localMaximum
    }

    private fun getOffsetPoint(pointIdxs: IntCoords, keypointId: Int, offsets: ThreeDFloatArray): FloatCoords {
        val (y, x) = pointIdxs
        val offsetY = offsets[y][x][keypointId]
        val offsetX = offsets[y][x][keypointId + partNames.size]
        return FloatCoords(offsetY, offsetX)
    }

    private fun getImageCoords(keypoint: Keypoint, offsets: ThreeDFloatArray): FloatCoords {
        // This is only invoked from keypoints that are on the
        // PriorityQueue, where x and y are integers, so it's safe to
        // round them (we can't cast float to int)
        val heatmapY = keypoint.y.roundToInt()
        val heatmapX = keypoint.x.roundToInt()
        val keypointId = keypoint.partId
        val (offsetY, offsetX) = getOffsetPoint(IntCoords(heatmapY, heatmapX), keypointId, offsets)
        val y = heatmapY * model.posenetOutputStride + offsetY
        val x = heatmapX * model.posenetOutputStride + offsetX

        return FloatCoords(y, x)
    }

    private fun squaredDistance(left: FloatCoords, right: FloatCoords): Float {
        val dy = left.y - right.y
        val dx = left.x - right.x
        return dy * dy + dx * dx
    }

    private fun withinNmsRadiusOfCorrespondingPoint(poses: List<Pose>,
                                                    squaredNmsRadius: Float,
                                                    point: FloatCoords,
                                                    keypointId: Int): Boolean {
        for (pose in poses) {
            val keypoints = pose.keypoints
            val sq = squaredDistance(point, keypoints[keypointId].coords)
            if (sq <= squaredNmsRadius)
                return true
        }

        return false
    }

    private fun traverseToTargetKeypoint(edgeId: Int,
                                         sourceKeypoint: Keypoint,
                                         targetKeypointId: Int,
                                         scores: ThreeDFloatArray,
                                         offsets: ThreeDFloatArray,
                                         displacements: ThreeDFloatArray): Keypoint {
        val height = scores.size
        val width = scores[0].size
        val sourceKeypointIndices = getStridedIndexNearPoint(sourceKeypoint.coords, height, width)

        val (displacementY, displacementX) = getDisplacement(edgeId, sourceKeypointIndices, displacements)
        val displacedPoint = FloatCoords(sourceKeypoint.y + displacementY, sourceKeypoint.x + displacementX)
        var targetKeypoint = displacedPoint

        val offsetRefineStep = 2
        for (i in 0 until offsetRefineStep) {
            val targetKPIdxs = getStridedIndexNearPoint(targetKeypoint, height, width)
            val (offsetY, offsetX) = getOffsetPoint(targetKPIdxs, targetKeypointId, offsets)

            targetKeypoint = FloatCoords(targetKPIdxs.y * model.posenetOutputStride + offsetY, targetKPIdxs.x * model.posenetOutputStride + offsetX)
        }

        val (targetKPIdxY, targetKPIdxX) = getStridedIndexNearPoint(targetKeypoint, height, width)
        val score = sigmoid(scores[targetKPIdxY][targetKPIdxX][targetKeypointId])
        return Keypoint(partNames[targetKeypointId], targetKeypointId, targetKeypoint, score)
    }

    private fun clamp(v: Int, min: Int, max: Int): Int {
        return when {
            v < min -> min
            v > max -> max
            else -> v
        }
    }

    private fun getStridedIndexNearPoint(kp: FloatCoords, height: Int, width: Int): IntCoords {
        val (_y, _x) = kp
        val y = clamp((_y / model.posenetOutputStride).roundToInt(), 0, height - 1)
        val x = clamp((_x / model.posenetOutputStride).roundToInt(), 0, width - 1)
        return IntCoords(y, x)
    }

    private fun getDisplacement(edgeId: Int, keypointIdxs: IntCoords, displacements: ThreeDFloatArray): FloatCoords {
        val numEdges = displacements[0][0].size / 2
        val (y, x) = keypointIdxs
        return FloatCoords(displacements[y][x][edgeId], displacements[y][x][edgeId + numEdges])
    }

    private fun getInstanceScore(keypoints: List<Keypoint>, existingPoses: List<Pose>, squaredNmsRadius: Int): Float {
        var scores = 0f
        for ((_, partId, coords, score) in keypoints) {
            if (withinNmsRadiusOfCorrespondingPoint(existingPoses, squaredNmsRadius.toFloat(), coords, partId))
                continue
            scores += score
        }

        return scores / partNames.size
    }

    private fun sigmoid(x: Float): Float {
        return (1.0f / (1.0f + exp(-1.0f * x)))
    }
}
