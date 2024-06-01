package com.dicoding.tfliteimageclassification

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import android.view.Surface
import androidx.camera.core.ImageProxy
import com.dicoding.util.Constants.MODEL_PATH
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.core.vision.ImageProcessingOptions
import org.tensorflow.lite.task.vision.classifier.Classifications
import org.tensorflow.lite.task.vision.classifier.ImageClassifier

class ImageClassifierHelper(
    var threshold: Float = 0.1f,
    var maxResults: Int = 3,
    var numThreads: Int = 4,
    val modelName: String = MODEL_PATH,
    val context: Context,
    val imageClassifierListener: ClassifierListener?
) {

    private var imageClassifier: ImageClassifier? = null

    //Listener = untuk memberitahu class uatama saat proses dilakukan berhasil/gagal
    //Apabila berhasil onResult yg dipanggil. error onError dipanggil. Disebut callback
    interface ClassifierListener {
        fun onError(error: String)
        fun onResults(
            results: List<Classifications>?,
            inferenceTime: Long
        )
    }

    init {
        setupImageClassifier()
    }

    //SetScoreThreshold = menentukan batas minimal keakuratan dri hasil yg ditampilkan 0.1 artinya 10%
    //SetMaxResult = menentukan batas maksimal jumlah hasil yg ditampilkan
    //setnumthreads = nentukan jmlah thread yg digunakan untuk pemrosesan ML
    //createFromFileandOprions = membuat imageClassifier berdasarkan asset file model dan option yg didefiniskan sblumnya
    fun setupImageClassifier() {
        val optionsBuilder = ImageClassifier.ImageClassifierOptions.builder()
            .setScoreThreshold(threshold)
            .setMaxResults(maxResults)

        val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)
        optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

        //Untuk mengecek instance ImageClassifier msih null/nggk
        //Tidak = lanjut ngubah image buffer jdi bitmap dg fungsi createMap
        //TensorImage dri bitmap tdi menggunakan fungsi Process
        try {
            imageClassifier =
                ImageClassifier.createFromFileAndOptions(context, modelName, optionsBuilder.build())
        } catch (e: IllegalStateException) {
//            imageClassifierListener?.onError(
//                "Image Classifier failed to initialize. See error logs details"
//            )
//            Log.e("ImageClassifierHelper", "TFLite failed to load model with error: " + e.message)
            e.printStackTrace()
        }
    }

    //Memanggil fungsi classify untuk memulai fungsi baru untuk mengonversi rotationDegress jdi orientation sperti contoh diatas
    //inferenceTIme untuk menghitung waktu dibutuhkan untuk melakukan pemrosesan
    fun classify(image: ImageProxy) {
        if (imageClassifier == null) {
            setupImageClassifier()
        }

        val bitmapBuffer = Bitmap.createBitmap(
            image.width,
            image.height,
            Bitmap.Config.ARGB_8888
        )

        image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }
        image.close()

        var inferenceTime = SystemClock.uptimeMillis()
        val imageProcessor = ImageProcessor.Builder().build()
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmapBuffer))

        val imageProcessingOptions = ImageProcessingOptions.builder()
            .setOrientation(getOrientationFromRotation(image.imageInfo.rotationDegrees))
            .build()

        val results = imageClassifier?.classify(tensorImage, imageProcessingOptions)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
        imageClassifierListener?.onResults(
            results, inferenceTime
        )
    }

    //ImageProcessingOptions = Mengatur orientasi gambar input sesuai dggambar yg ada pada model
    //setOrientation = fungsi untuk disediakan
    //rotation degress = jai orientation sperti contoh diatas
    //Habis siap nanti msuk ke classify
    private fun getOrientationFromRotation(rotation: Int): ImageProcessingOptions.Orientation {
        return when (rotation) {
            Surface.ROTATION_270 -> ImageProcessingOptions.Orientation.BOTTOM_RIGHT
            Surface.ROTATION_180 -> ImageProcessingOptions.Orientation.RIGHT_BOTTOM
            Surface.ROTATION_90 -> ImageProcessingOptions.Orientation.TOP_LEFT
            else -> ImageProcessingOptions.Orientation.RIGHT_TOP
        }
    }
}


