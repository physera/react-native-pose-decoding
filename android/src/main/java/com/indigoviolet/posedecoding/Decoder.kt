package com.indigoviolet.posedecoding

import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import android.util.Log
import java.nio.ByteBuffer
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteOrder
import java.lang.reflect.Array
import java.nio.channels.FileChannel
import java.util.*

data class TfliteModel @JvmOverloads constructor(val modelFd: AssetFileDescriptor,
                                                 val useNNAPI: Boolean,
                                                 val useGpuDelegate: Boolean,
                                                 val allowFp16Precision: Boolean,
                                                 val numThreads: Int,
                                                 val posenetOutputStride: Int = 16) {

    private val gpuDelegate: GpuDelegate? = if (useGpuDelegate) GpuDelegate() else null
    val tflite: Interpreter
    private val inputTensor: Tensor

    private val modelFileBuffer: ByteBuffer
        get() {
            val inputStream = FileInputStream(modelFd.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = modelFd.startOffset
            val declaredLength = modelFd.declaredLength
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }

    init {
        tflite = Interpreter(modelFileBuffer, getOptions())
        inputTensor = tflite.getInputTensor(0)
    }

    val inputSize: Int
        get() = inputTensor.shape()[1]

    val inputChannels: Int
        get() = inputTensor.shape()[3]

    val bytesPerChannel: Int
        get() = inputTensor.dataType().byteSize()

    init {
        Log.i("ReactNative", "inputSize=$inputSize; inputChannels=$inputChannels, bytesPerChannel=$bytesPerChannel")
    }

    private fun getOptions(): Interpreter.Options {
        val options = (Interpreter.Options()).setUseNNAPI(useNNAPI).setAllowFp16PrecisionForFp32(allowFp16Precision).setNumThreads(numThreads)
        if (useGpuDelegate) {
            options.addDelegate(gpuDelegate)
        }
        return options
    }

    fun close() {
        tflite.close()
        gpuDelegate?.close()
    }

}

data class DecoderParams @JvmOverloads constructor(val threshold: Float = 0f,
                                                   val posenetNumResults: Int = 1,
                                                   val posenetNMSRadius: Int = 20,
                                                   val posenetMean: Float = 128.0f,
                                                   val posenetStd: Float = 128.0f)

abstract class Decoder(private val model: TfliteModel, private val params: DecoderParams) {

    private val inputTensorSize = model.inputSize * model.inputSize * model.inputChannels * model.bytesPerChannel
    protected val inputBuf: ByteBuffer = ByteBuffer.allocateDirect(inputTensorSize)
    init {
        inputBuf.order(ByteOrder.nativeOrder())
    }

    protected val outputBuf: MutableMap<Int, Any> = mutableMapOf()

    init {
        for (i in 0 until model.tflite.outputTensorCount) {
            val shape = model.tflite.getOutputTensor(i).shape()
            Log.i("ReactNative", "outputTensor $i of ${model.tflite.outputTensorCount} has shape ${shape.joinToString(", ")}")

            // Make a multi-dimensional array dynamically
            outputBuf[i] = Array.newInstance(Float::class.java, *shape)
        }
    }

    var timing: MutableMap<String, Long>? = null

    fun run(bitmap: Bitmap): Long {
        getImageData(bitmap)
        timing!!["inferenceBeginTime"] = Calendar.getInstance().timeInMillis
        model.tflite.runForMultipleInputsOutputs(arrayOf(inputBuf), outputBuf)
        timing!!["inferenceEndTime"] = Calendar.getInstance().timeInMillis
        return model.tflite.lastNativeInferenceDurationNanoseconds
    }

    fun getPoses(): List<Map<String, Any>> = decode().map { it.toMap() }

    private val pixelBuf = IntArray(model.inputSize * model.inputSize)

    private fun getImageData(bitmap: Bitmap) {
        timing!!["imageDataBeginTime"] = Calendar.getInstance().timeInMillis
        inputBuf.rewind()
        bitmap.getPixels(pixelBuf, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var pixel = 0
        for (i in 0 until model.inputSize) {
            for (j in 0 until model.inputSize) {
                val pixelValue = pixelBuf[pixel++]
                addPixelValue(pixelValue)
            }
        }
        inputBuf.rewind()
        timing!!["imageDataEndTime"] = Calendar.getInstance().timeInMillis
    }

    abstract fun decode(): List<Pose>
    abstract fun addPixelValue(v: Int)
}
