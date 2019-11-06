package com.indigoviolet.posedecoding

abstract class Decoder {
    abstract fun decode(outputBuf: Map<Int, Any>): List<Map<String, Any>>
    // abstract fun getImageData()
}
