package edu.ufl.absent

import android.content.Context
import com.qualcomm.qti.QnnDelegate
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.gpu.GpuDelegateFactory
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.channels.FileChannel

enum class Backend { NPU, GPU, CPU }

/**
 * A float TFLite model on LiteRT's classic Interpreter with one backend: Qualcomm's QNN delegate
 * on the Hexagon NPU (HTP, fp16), the GPU delegate, or XNNPACK on the CPU. Inputs and outputs
 * are direct float buffers in model order.
 */
class TfliteModel private constructor(
    private val interpreter: Interpreter,
    private val delegate: Delegate?,
    val backend: Backend,
) : AutoCloseable {
    private val inputBytes = (0 until interpreter.inputTensorCount).map { direct(interpreter.getInputTensor(it).numElements()) }
    private val outputBytes = (0 until interpreter.outputTensorCount).map { direct(interpreter.getOutputTensor(it).numElements()) }
    val inputs: List<FloatBuffer> = inputBytes.map { it.asFloatBuffer() }
    val outputs: List<FloatBuffer> = outputBytes.map { it.asFloatBuffer() }
    fun inputName(index: Int): String = interpreter.getInputTensor(index).name()

    fun run() {
        inputs.forEach { it.rewind() }
        val inArray = inputBytes.map { it.rewind() }.toTypedArray<Any>()
        val outMap = HashMap<Int, Any>()
        outputBytes.forEachIndexed { i, bytes -> outMap[i] = bytes.rewind() }
        interpreter.runForMultipleInputsOutputs(inArray, outMap)
        outputs.forEach { it.rewind() }
    }

    override fun close() {
        interpreter.close()
        (delegate as? AutoCloseable)?.close()
    }

    private fun direct(count: Int): ByteBuffer = ByteBuffer.allocateDirect(count * 4).order(ByteOrder.nativeOrder())

    companion object {
        fun load(context: Context, file: File, backend: Backend): TfliteModel {
            val mapped = FileInputStream(file).channel.use { it.map(FileChannel.MapMode.READ_ONLY, 0, it.size()) }
            // The NPU cache is keyed by this name, so a replaced model file gets a fresh compile.
            return create(context, mapped, backend, "${file.name}_${file.length()}_${file.lastModified()}")
        }

        fun loadAsset(context: Context, asset: String, backend: Backend): TfliteModel {
            val fd = context.assets.openFd(asset)
            val mapped = FileInputStream(fd.fileDescriptor).channel.use { it.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength) }
            return create(context, mapped, backend, asset)
        }

        private fun create(context: Context, model: ByteBuffer, backend: Backend, name: String): TfliteModel {
            val options = Interpreter.Options()
            if (backend == Backend.NPU) {
                // FastRPC looks for the Hexagon-side (skel) libraries on this path.
                val dirs = listOf(context.applicationInfo.nativeLibraryDir, "/vendor/lib/rfsa/adsp", "/vendor/dsp/cdsp", "/vendor/dsp", "/system/lib/rfsa/adsp", "/dsp")
                android.system.Os.setenv("ADSP_LIBRARY_PATH", dirs.joinToString(";"), true)
                System.loadLibrary("qnn_delegate_jni")
                android.util.Log.i("TfliteModel", "QNN ${QnnDelegate.getVersion().joinToString(".")}: " +
                    QnnDelegate.Capability.values().joinToString { "$it=${runCatching { QnnDelegate.checkCapability(it) }.getOrNull()}" })
            }
            val delegate: Delegate? = when (backend) {
                Backend.NPU -> QnnDelegate(QnnDelegate.Options().apply {
                    setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
                    setSkelLibraryDir(context.applicationInfo.nativeLibraryDir)
                    setHtpPerformanceMode(QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_BURST)
                    setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16)
                    // App processes may only open an unsigned protection domain on the DSP.
                    setHtpPdSession(QnnDelegate.Options.HtpPdSession.HTP_PD_SESSION_UNSIGNED)
                    setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_INFO)
                    // The compiled NPU graph is cached, so later launches skip compilation.
                    setCacheDir(context.cacheDir.absolutePath)
                    setModelToken(name.replace(Regex("[^A-Za-z0-9_]"), "_"))
                })
                Backend.GPU -> GpuDelegate(GpuDelegateFactory.Options()
                    .setPrecisionLossAllowed(true)
                    .setInferencePreference(GpuDelegateFactory.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED))
                Backend.CPU -> null
            }
            if (delegate != null) options.addDelegate(delegate) else options.setNumThreads(4)
            return try {
                TfliteModel(Interpreter(model, options), delegate, backend)
            } catch (e: Exception) {
                (delegate as? AutoCloseable)?.close()
                throw e
            }
        }
    }
}
