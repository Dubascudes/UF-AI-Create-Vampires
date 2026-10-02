package edu.ufl.absent

import android.content.Context
import android.media.Image
import android.util.Log
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Person confidence (0–255), rows top-down, in upright (rotated) image orientation, plus each
 * detected person's box as (cx, cy, w, h) normalised to the upright image.
 */
class Mask(val bytes: ByteArray, val width: Int, val height: Int, val boxes: List<FloatArray>) {
    val people: Int get() = boxes.size
}

/** Where person fills come from: the laptop over USB, or a model on the phone. */
interface FillSource {
    val label: String
    val roundTripMs: Samples
    val completed: Int
    val lastError: String?
    val lastCrop: String
    /**
     * Blocks until the source is free, then fills [mask]'s people in [frame] asynchronously and
     * calls [done] with the result (null when there is nothing to fill or it failed). [frame]
     * must be a snapshot the caller no longer touches.
     */
    fun submit(frame: YuvFrame, rotation: Int, mask: Mask, done: (Fill?) -> Unit)
    fun close()
}

/** One camera frame's results, all computed from that frame: its mask and, if anyone was in view, its fill. */
class Processed(val timestamp: Long, val mask: Mask, val fill: Fill?)

/** A YOLO segmentation export: input is NCHW RGB 0–1 at width×height (upright, portrait). */
data class ModelSpec(val asset: String, val width: Int, val height: Int, val label: String)

val MODELS = listOf(
    ModelSpec("yolo26n-seg-384x672.tflite", 384, 672, "yolo 384×672"),
    ModelSpec("yolo26n-seg-480x640.tflite", 480, 640, "yolo 480×640"),
    ModelSpec("yolo26n-seg-288x384.tflite", 288, 384, "yolo 288×384"),
)

/**
 * YOLO26n-seg on one worker thread through LiteRT (GPU, CPU fallback).
 * The GL thread copies each camera image and hands it off; frames arriving while
 * the worker is busy are dropped, so the mask never queues behind the camera.
 */
class PersonMasker(private val context: Context, val spec: ModelSpec) {
    var delegate = "loading"
        private set
    val convertMs = Samples()
    val inferMs = Samples()
    val decodeMs = Samples()
    val totalMs = Samples()
    /** Receives every processed frame, from the mask worker or the fill worker. */
    @Volatile var sink: (Processed) -> Unit = {}
    @Volatile var masksCompleted = 0
        private set
    /** When set, each mask is also offered, with its frame, for remote inpainting. */
    @Volatile var inpainter: FillSource? = null
    private val busy = AtomicBoolean(false)
    private val worker = Executors.newSingleThreadExecutor()
    private var model: TfliteModel? = null
    private val yuv = YuvFrame()
    private val input = FloatArray(3 * spec.width * spec.height)
    private var det = FloatArray(0)
    private var proto = FloatArray(0)

    init {
        worker.execute { load() }
    }

    private fun load() {
        for (backend in listOf(Backend.GPU, Backend.CPU)) {
            try {
                model = TfliteModel.loadAsset(context, spec.asset, backend)
                delegate = backend.name
                return
            } catch (e: Exception) {
                Log.w(TAG, "LiteRT $backend failed for ${spec.asset}", e)
            }
        }
        delegate = "failed"
    }

    /**
     * Called on the GL thread. Copies [image] and returns true if the worker is free (the caller
     * then keeps that frame's camera texture for display); the caller closes the image.
     */
    fun offer(image: Image, rotation: Int, timestamp: Long): Boolean {
        if (model == null || !busy.compareAndSet(false, true)) return false
        yuv.copy(image)
        worker.execute {
            try {
                run(rotation, timestamp)
            } catch (e: Exception) {
                Log.w(TAG, "segmentation failed", e)
            } finally {
                busy.set(false)
            }
        }
        return true
    }

    private fun run(rotation: Int, timestamp: Long) {
        val t0 = System.nanoTime()
        fillInput(rotation)
        val t1 = System.nanoTime()
        val m = model!!
        m.inputs[0].put(input)
        m.run()
        // Output order is not guaranteed; the prototype tensor is the one with 32 × (h/4) × (w/4) values.
        val protoSize = 32 * (spec.width / 4) * (spec.height / 4)
        val (protoBuffer, detBuffer) = if (m.outputs[1].capacity() == protoSize) Pair(m.outputs[1], m.outputs[0]) else Pair(m.outputs[0], m.outputs[1])
        if (det.size != detBuffer.capacity()) { det = FloatArray(detBuffer.capacity()); proto = FloatArray(protoBuffer.capacity()) }
        detBuffer.get(det); protoBuffer.get(proto)
        val t2 = System.nanoTime()
        val mask = decode(det, proto)
        val t3 = System.nanoTime()
        masksCompleted++
        convertMs.add((t1 - t0) / 1e6); inferMs.add((t2 - t1) / 1e6); decodeMs.add((t3 - t2) / 1e6); totalMs.add((t3 - t0) / 1e6)
        val source = inpainter
        if (mask.people > 0 && source != null) {
            // Waits while the fill worker is busy with the previous frame, so this worker masks
            // the next frame while that one is being filled, and no frame is shown unfilled.
            source.submit(yuv.snapshot(), rotation, mask) { fill -> sink(Processed(timestamp, mask, fill)) }
        } else {
            sink(Processed(timestamp, mask, null))
        }
    }

    /**
     * YUV→RGB into the planar model input, rotated clockwise by [rotation] so people stand
     * upright, with nearest-neighbour scaling from the upright camera image to the model size.
     */
    private fun fillInput(rotation: Int) {
        val plane = spec.width * spec.height
        val layout = Layout(rotation, yuv.width, yuv.height, yuv.yRow, yuv.uvRow, yuv.uvPixel)
        if (layout != tableLayout) buildTables(layout)
        val yIndex = yTable
        val uvIndex = uvTable
        val y = yuv.y; val u = yuv.u; val v = yuv.v
        for (i in 0 until plane) {
            val c = 1192 * max((y[yIndex[i]].toInt() and 0xff) - 16, 0)
            val uv = uvIndex[i]
            val cu = (u[uv].toInt() and 0xff) - 128
            val cv = (v[uv].toInt() and 0xff) - 128
            input[i] = clampByte((c + 1634 * cv) shr 10) * INV_255
            input[plane + i] = clampByte((c - 833 * cv - 400 * cu) shr 10) * INV_255
            input[2 * plane + i] = clampByte((c + 2066 * cu) shr 10) * INV_255
        }
    }

    private data class Layout(val rotation: Int, val width: Int, val height: Int, val yRow: Int, val uvRow: Int, val uvPixel: Int)
    private var tableLayout: Layout? = null
    private var yTable = IntArray(0)
    private var uvTable = IntArray(0)

    /** Source Y and UV indices for every model input pixel; rebuilt only when the camera layout changes. */
    private fun buildTables(layout: Layout) {
        val w = spec.width
        val h = spec.height
        val swap = layout.rotation == 90 || layout.rotation == 270
        val uprightW = if (swap) layout.height else layout.width
        val uprightH = if (swap) layout.width else layout.height
        yTable = IntArray(w * h)
        uvTable = IntArray(w * h)
        for (dy in 0 until h) {
            val uy = dy * uprightH / h
            for (dx in 0 until w) {
                val ux = dx * uprightW / w
                val sx: Int
                val sy: Int
                when (layout.rotation) {
                    90 -> { sx = uy; sy = layout.height - 1 - ux }
                    180 -> { sx = layout.width - 1 - ux; sy = layout.height - 1 - uy }
                    270 -> { sx = layout.width - 1 - uy; sy = ux }
                    else -> { sx = ux; sy = uy }
                }
                yTable[dy * w + dx] = sy * layout.yRow + sx
                uvTable[dy * w + dx] = (sy shr 1) * layout.uvRow + (sx shr 1) * layout.uvPixel
            }
        }
        tableLayout = layout
    }

    private fun clampByte(value: Int): Int = if (value < 0) 0 else if (value > 255) 255 else value

    /**
     * Raw YOLO head: det is [116][n] (cx, cy, w, h normalised to 0–1, 80 class scores, 32 mask
     * coefficients); proto is [32][h/4][w/4]. Keeps people, suppresses overlapping boxes, and
     * takes the per-pixel maximum of each person's mask inside its box.
     */
    private fun decode(det: FloatArray, proto: FloatArray): Mask {
        val n = det.size / 116
        val candidates = (0 until n).filter { det[4 * n + it] > SCORE_THRESHOLD }.sortedByDescending { det[4 * n + it] }
        val kept = ArrayList<Int>()
        for (i in candidates) {
            if (kept.size >= MAX_PEOPLE) break
            if (kept.none { iou(det, n, i, it) > IOU_THRESHOLD }) kept.add(i)
        }
        val pw = spec.width / 4
        val ph = spec.height / 4
        val plane = pw * ph
        val mask = FloatArray(plane)
        val coeff = FloatArray(32)
        for (i in kept) {
            for (k in 0 until 32) coeff[k] = det[(84 + k) * n + i]
            val cx = det[i]; val cy = det[n + i]; val bw = det[2 * n + i]; val bh = det[3 * n + i]
            val x0 = max(0, ((cx - bw / 2) * pw).toInt()); val x1 = min(pw, ((cx + bw / 2) * pw).toInt() + 1)
            val y0 = max(0, ((cy - bh / 2) * ph).toInt()); val y1 = min(ph, ((cy + bh / 2) * ph).toInt() + 1)
            for (y in y0 until y1) for (x in x0 until x1) {
                val p = y * pw + x
                var sum = 0f
                for (k in 0 until 32) sum += coeff[k] * proto[k * plane + p]
                val value = 1f / (1f + exp(-sum))
                if (value > mask[p]) mask[p] = value
            }
        }
        val boxes = kept.map { floatArrayOf(det[it], det[n + it], det[2 * n + it], det[3 * n + it]) }
        return Mask(dilate(mask, pw, ph), pw, ph, boxes)
    }

    /** Square max filter of radius [DILATE] (in mask pixels, 4 camera pixels each), so person edges are covered. */
    private fun dilate(mask: FloatArray, w: Int, h: Int): ByteArray {
        val rows = FloatArray(mask.size)
        for (y in 0 until h) for (x in 0 until w) {
            var m = 0f
            for (dx in max(0, x - DILATE)..min(w - 1, x + DILATE)) m = max(m, mask[y * w + dx])
            rows[y * w + x] = m
        }
        return ByteArray(mask.size) { i ->
            val x = i % w; val y = i / w
            var m = 0f
            for (dy in max(0, y - DILATE)..min(h - 1, y + DILATE)) m = max(m, rows[dy * w + x])
            (m * 255f).toInt().toByte()
        }
    }

    private fun iou(det: FloatArray, n: Int, a: Int, b: Int): Float {
        val ax0 = det[a] - det[2 * n + a] / 2; val ax1 = det[a] + det[2 * n + a] / 2
        val ay0 = det[n + a] - det[3 * n + a] / 2; val ay1 = det[n + a] + det[3 * n + a] / 2
        val bx0 = det[b] - det[2 * n + b] / 2; val bx1 = det[b] + det[2 * n + b] / 2
        val by0 = det[n + b] - det[3 * n + b] / 2; val by1 = det[n + b] + det[3 * n + b] / 2
        val inter = max(0f, min(ax1, bx1) - max(ax0, bx0)) * max(0f, min(ay1, by1) - max(ay0, by0))
        val union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
        return if (union > 0f) inter / union else 0f
    }

    fun close() {
        worker.execute {
            model?.close(); model = null
        }
        worker.shutdown()
    }

    companion object {
        private const val TAG = "PersonMasker"
        private const val INV_255 = 1f / 255f
        private const val SCORE_THRESHOLD = 0.3f
        private const val IOU_THRESHOLD = 0.5f
        private const val MAX_PEOPLE = 20
        private const val DILATE = 1
    }
}

/** Rolling median over the last 120 samples. */
class Samples {
    private val values = ArrayDeque<Double>()
    @Synchronized fun add(value: Double) {
        values.addLast(value)
        if (values.size > 120) values.removeFirst()
    }
    @Synchronized fun median(): Double? = if (values.isEmpty()) null else values.sorted()[values.size / 2]
}
