package com.indigoviolet.posedecoding

import android.util.Log

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.android.OpenCVLoader
import kotlin.math.exp

object CPMHourglassDecoder {

    private val partNames = arrayOf("top", "neck",
                                   "rightShoulder", "rightElbow", "rightWrist",
                                   "leftShoulder", "leftElbow", "leftWrist",
                                   "rightHip", "rightKnee", "rightAnkle",
                                   "leftHip", "leftKnee", "leftAnkle")

    init {
        //        System.loadLibrary("opencv_java");
        System.loadLibrary("opencv_java3")
        if (!OpenCVLoader.initDebug()) {
            Log.i("ReactNative", "opencv did not initialize")
        }
    }

    @JvmStatic @JvmOverloads fun decode(outputMap: Map<Int, Any>, threshold: Float = 0f): List<Map<String, Any>> {

        // 1x96x96x14 (CPM) or 1x48x48x14 (Hourglass)
        @Suppress("UNCHECKED_CAST")
        val heatMapArray = (outputMap[0] as Array<Array<Array<FloatArray>>>)[0]
        val outputW = heatMapArray.size
        val outputH = heatMapArray[0].size

        val keypoints = mutableListOf<Keypoint>()

        // Gaussian Filter 5*5
        val mMat = Mat(outputW, outputH, CvType.CV_32F)

        val tempArray = FloatArray(outputW * outputH)
        val outTempArray = FloatArray(outputW * outputH)
        var score = 0f
        for (i in 0..13) {
            var index = 0
            for (x in 0 until outputW) {
                for (y in 0 until outputH) {
                    tempArray[index] = heatMapArray[y][x][i]
                    index++
                }
            }

            mMat.put(0, 0, tempArray)
            Imgproc.GaussianBlur(mMat, mMat, Size(5.0, 5.0), 0.0, 0.0)
            mMat.get(0, 0, outTempArray)

            var maxX = 0f
            var maxY = 0f
            var max = 0f

            // Find keypoint coordinate through maximum values
            for (x in 0 until outputW) {
                for (y in 0 until outputH) {
                    val center = get(x, y, outTempArray, outputW, outputH)
                    if (center > max) {
                        max = center
                        maxX = x.toFloat() / 0.25f //looks like this should be outputW/inputW
                        maxY = y.toFloat() / 0.25f
                    }
                }
            }

            val s = max // sigmoid(max)
            if (s <= threshold) {
                continue
            }


            // Log.i("ReactNative", "before: ${partNames[i]} ${tempArray.filter { it > 0 }.joinToString()}")
            // Log.i("ReactNative", "after: ${partNames[i]} ${outTempArray.filter { it > 0 }.joinToString()}")
            Log.i("ReactNative", "${partNames[i]} score=$s, maxX=$maxX, maxY=$maxY")
            keypoints.add(Keypoint(partNames[i], i, FloatCoords(maxY, maxX), s))
            score += s
        }

        return listOf(Pose(score, keypoints).toMap())
    }

    private fun sigmoid(x: Float): Float {
        return (1.0 / (1.0 + exp(-1.0 * x))).toFloat()
    }

    private fun get(
        x: Int,
        y: Int,
        arr: FloatArray,
        outputW: Int,
        outputH: Int
    ): Float {
        return if (x < 0 || y < 0 || x >= outputW || y >= outputH) -1f else arr[x * outputW + y]
    }
}
