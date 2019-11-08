package com.indigoviolet.posedecoding

import android.util.Log
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.exp

class CPMHourglassDecoder private constructor(
    private val model: TfliteModel,
    private val params: DecoderParams
) : Decoder(model, params) {

    companion object {
        private lateinit var instance: CPMHourglassDecoder

        @JvmStatic
        fun getInstance(model: TfliteModel, params: DecoderParams): CPMHourglassDecoder {
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
                instance = CPMHourglassDecoder(model, params)
            }
            return instance
        }
    }

    private val partNames = arrayOf("top", "neck",
            "rightShoulder", "rightElbow", "rightWrist",
            "leftShoulder", "leftElbow", "leftWrist",
            "rightHip", "rightKnee", "rightAnkle",
            "leftHip", "leftKnee", "leftAnkle")

    init {
        System.loadLibrary("opencv_java3")
        if (!OpenCVLoader.initDebug()) {
            Log.i("ReactNative", "opencv did not initialize")
        }
    }

    override fun addPixelValue(v: Int) {
        // bgr
        inputBuf.putFloat((v and 0xFF).toFloat())
        inputBuf.putFloat(((v shr 8) and 0xFF).toFloat())
        inputBuf.putFloat((((v shr 16) and 0xFF)).toFloat())
    }

    override fun decode(): List<Pose> {

        // 1x96x96x14 (CPM) or 1x48x48x14 (Hourglass)
        @Suppress("UNCHECKED_CAST")
        val heatMapArray = (outputBuf[0] as Array<Array<Array<FloatArray>>>)[0]
        val outputW = heatMapArray.size

        val keypoints = mutableListOf<Keypoint>()

        // Gaussian Filter 5*5
        val mMat = Mat(outputW, outputW, CvType.CV_32F)

        val tempArray = FloatArray(outputW * outputW)
        var score = 0f
        for (i in partNames.indices) {

            // unwrap heatmapArray into tempArray
            var index = 0
            for (x in 0 until outputW) {
                for (y in 0 until outputW) {
                    tempArray[index] = heatMapArray[y][x][i]
                    index++
                }
            }

            // Much faster to put in and read out whole vectors than
            // one element at a time, because opencv's internal
            // representation is consecutive
            mMat.put(0, 0, tempArray)
            Imgproc.GaussianBlur(mMat, mMat, Size(5.0, 5.0), 0.0, 0.0)
            mMat.get(0, 0, tempArray)

            var maxX = 0f
            var maxY = 0f
            var max = 0f
            val ratio = outputW.toFloat() / model.inputSize

            // Find keypoint coordinate through maximum values
            for (x in 0 until outputW) {
                for (y in 0 until outputW) {
                    val center = tempArray[x * outputW + y]
                    if (center > max) {
                        max = center
                        maxX = x.toFloat() / ratio
                        maxY = y.toFloat() / ratio
                    }
                }
            }

             if (max <= params.threshold) {
                continue
            }

            keypoints.add(Keypoint(partNames[i], i, FloatCoords(maxY, maxX), max))
            score += max
        }

        return listOf(Pose(score, keypoints))
    }
}
