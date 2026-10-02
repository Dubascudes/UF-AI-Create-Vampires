package edu.ufl.absent

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.max
import kotlin.math.min

/**
 * LaMa-Dilated with the LaMa authors' Places2 (scene) weights, exported by
 * scripts/export-lama-dilated-places.py with the same contract as Qualcomm's AI Hub export
 * (NHWC 512×512 RGB 0–1 plus a 0/1 hole mask in, filled image out). Qualcomm's own export uses
 * the CelebA-HQ face checkpoint, which paints faces into holes. Each person or group of nearby
 * people gets its own crop at native resolution where possible, so holes stay small relative to
 * their context. Up to [Fill.MAX_CROPS] crops per frame, packed into one 1024×1024 atlas. The
 * model is read from the app's external files directory:
 * `adb push lama_dilated_places.tflite /sdcard/Android/data/edu.ufl.absent/files/`.
 */
class LocalInpainter(private val context: Context, private val backend: Backend) : FillSource {
    override val label = "phone ${backend.name}"
    /** Whole fill for one frame (all crops). */
    override val roundTripMs = Samples()
    val prepMs = Samples()
    val modelMs = Samples()
    @Volatile override var completed = 0
        private set
    @Volatile override var lastError: String? = "loading"
        private set
    @Volatile override var lastCrop = "—"
        private set
    @Volatile var loadedAs = "loading"
        private set
    private val worker = Executors.newSingleThreadExecutor()
    private val modelFile = File(context.getExternalFilesDir(null), MODEL_FILE)
    private var model: TfliteModel? = null
    private var imageInput = 0
    private var painted = FloatArray(SIZE * SIZE * 3)
    private val image = FloatArray(SIZE * SIZE * 3)
    private val hole = FloatArray(SIZE * SIZE)
    private val holeRows = FloatArray(SIZE * SIZE)

    init {
        worker.execute { load() }
    }

    private fun load() {
        if (!modelFile.isFile) {
            lastError = "missing ${modelFile.path}"
            loadedAs = "missing"
            return
        }
        val start = System.nanoTime()
        try {
            var m = TfliteModel.load(context, modelFile, backend)
            // The QNN delegate compiles the graph for the NPU (~15 s) and writes its cache file
            // only when closed, so after a compile (a slow load) close and reopen once to
            // persist it. Later launches restore the cache in well under a second.
            if (backend == Backend.NPU && System.nanoTime() - start > 3_000_000_000L) {
                m.close()
                m = TfliteModel.load(context, modelFile, backend)
            }
            val loadMs = (System.nanoTime() - start) / 1_000_000
            imageInput = (0 until m.inputs.size).firstOrNull { m.inputName(it).contains("image") } ?: 0
            // Benchmark on a blank frame, so the HUD shows model speed before anyone is in view.
            val runs = (0 until 3).map { invoke(m) }
            model = m
            loadedAs = "${backend.name} load $loadMs ms, bench ${runs.joinToString("/") { "%.0f".format(it) }} ms"
            Log.i(TAG, "LaMa-Dilated $loadedAs")
            lastError = null
        } catch (e: Throwable) {
            Log.w(TAG, "LaMa-Dilated failed on $backend", e)
            loadedAs = "failed"
            lastError = "$backend: ${e.message?.take(80)}"
        }
    }

    /** Runs the model on the current [image] and [hole]; the result lands in [painted]. Returns ms. */
    private fun invoke(m: TfliteModel): Double {
        val t = System.nanoTime()
        m.inputs[imageInput].put(image)
        m.inputs[1 - imageInput].put(hole)
        m.run()
        m.outputs[0].get(painted)
        return (System.nanoTime() - t) / 1e6
    }

    private val slot = java.util.concurrent.Semaphore(1)

    override fun submit(frame: YuvFrame, rotation: Int, mask: Mask, done: (Fill?) -> Unit) {
        slot.acquire()
        worker.execute {
            var fill: Fill? = null
            try {
                fill = if (model != null) fill(frame, rotation, mask) else null
            } catch (e: Exception) {
                lastError = e.javaClass.simpleName
                Log.w(TAG, "local inpaint failed", e)
            } finally {
                slot.release()
                done(fill)
            }
        }
    }

    private fun fill(frame: YuvFrame, rotation: Int, mask: Mask): Fill? {
        val start = System.nanoTime()
        val fullW = frame.uprightWidth(rotation)
        val fullH = frame.uprightHeight(rotation)
        val crops = crops(mask, fullW, fullH)
        if (crops.isEmpty()) return null
        lastCrop = crops.joinToString(" ") { "${it[2]}×${it[3]}" }

        val atlas = IntArray(ATLAS * ATLAS)
        val rects = FloatArray(4 * Fill.MAX_CROPS)
        val atlasRects = FloatArray(4 * Fill.MAX_CROPS)
        var prep = 0L
        var run = 0L
        for ((i, crop) in crops.withIndex()) {
            val (x0, y0, w, h) = crop
            val scale = SIZE.toFloat() / max(w, h)
            val cw = min(SIZE, (w * scale).toInt())
            val ch = min(SIZE, (h * scale).toInt())
            val t0 = System.nanoTime()
            prepare(frame, rotation, mask, x0, y0, w, h, cw, ch, fullW, fullH)
            val t1 = System.nanoTime()
            invoke(model!!)
            val t2 = System.nanoTime()
            // Slot i of the 2×2 atlas; only the content region (the rest is padding) is kept.
            val ax = (i % 2) * SIZE
            val ay = (i / 2) * SIZE
            for (y in 0 until ch) for (x in 0 until cw) {
                val p = (y * SIZE + x) * 3
                val r = (painted[p] * 255f).toInt().coerceIn(0, 255)
                val g = (painted[p + 1] * 255f).toInt().coerceIn(0, 255)
                val b = (painted[p + 2] * 255f).toInt().coerceIn(0, 255)
                atlas[(ay + y) * ATLAS + ax + x] = (0xff shl 24) or (r shl 16) or (g shl 8) or b
            }
            floatArrayOf(x0.toFloat() / fullW, y0.toFloat() / fullH, w.toFloat() / fullW, h.toFloat() / fullH).copyInto(rects, 4 * i)
            floatArrayOf(ax.toFloat() / ATLAS, ay.toFloat() / ATLAS, cw.toFloat() / ATLAS, ch.toFloat() / ATLAS).copyInto(atlasRects, 4 * i)
            prep += (t1 - t0) + (System.nanoTime() - t2)
            run += t2 - t1
        }
        val bitmap = Bitmap.createBitmap(atlas, ATLAS, ATLAS, Bitmap.Config.ARGB_8888)
        completed++
        prepMs.add(prep / 1e6)
        modelMs.add(run / 1e6 / crops.size)
        roundTripMs.add((System.nanoTime() - start) / 1e6)
        return Fill(bitmap, crops.size, rects, atlasRects)
    }

    /**
     * Model input for one crop: the crop scaled by cw/w into the top-left of 512×512, edge pixels
     * repeated into the padding; hole = person mask (bilinear from the model mask, > 0.5), grown by
     * [HOLE_GROW] pixels so a person's blurred outline is not copied into the fill.
     */
    private fun prepare(frame: YuvFrame, rotation: Int, mask: Mask, x0: Int, y0: Int, w: Int, h: Int, cw: Int, ch: Int, fullW: Int, fullH: Int) {
        val sx = w.toFloat() / cw
        val sy = h.toFloat() / ch
        for (y in 0 until SIZE) {
            val cy = min(y, ch - 1)
            val uy = min(fullH - 1, y0 + ((cy + 0.5f) * sy).toInt())
            val inRows = y < ch
            for (x in 0 until SIZE) {
                val cx = min(x, cw - 1)
                val ux = min(fullW - 1, x0 + ((cx + 0.5f) * sx).toInt())
                val argb = frame.argbUpright(ux, uy, rotation)
                val p = (y * SIZE + x) * 3
                image[p] = ((argb shr 16) and 0xff) * INV_255
                image[p + 1] = ((argb shr 8) and 0xff) * INV_255
                image[p + 2] = (argb and 0xff) * INV_255
                holeRows[y * SIZE + x] = if (inRows && x < cw && sampleMask(mask, (ux + 0.5f) / fullW, (uy + 0.5f) / fullH) > 127) 1f else 0f
            }
        }
        // Separable square max filter.
        for (y in 0 until SIZE) for (x in 0 until SIZE) {
            var m = 0f
            for (dx in max(0, x - HOLE_GROW)..min(SIZE - 1, x + HOLE_GROW)) if (holeRows[y * SIZE + dx] > m) { m = 1f; break }
            hole[y * SIZE + x] = m
        }
        for (x in 0 until SIZE) for (y in 0 until SIZE) {
            var m = 0f
            for (dy in max(0, y - HOLE_GROW)..min(SIZE - 1, y + HOLE_GROW)) if (hole[dy * SIZE + x] > m) { m = 1f; break }
            holeRows[y * SIZE + x] = m
        }
        holeRows.copyInto(hole)
    }

    private fun sampleMask(mask: Mask, u: Float, v: Float): Int {
        val x = (u * mask.width - 0.5f).coerceIn(0f, mask.width - 1f)
        val y = (v * mask.height - 0.5f).coerceIn(0f, mask.height - 1f)
        val x0 = x.toInt(); val y0 = y.toInt()
        val x1 = min(x0 + 1, mask.width - 1); val y1 = min(y0 + 1, mask.height - 1)
        val fx = x - x0; val fy = y - y0
        fun at(px: Int, py: Int) = (mask.bytes[py * mask.width + px].toInt() and 0xff).toFloat()
        return ((at(x0, y0) * (1 - fx) + at(x1, y0) * fx) * (1 - fy) + (at(x0, y1) * (1 - fx) + at(x1, y1) * fx) * fy).toInt()
    }

    /**
     * Crop rectangles (x, y, w, h) in upright pixels: each person's box plus a context margin,
     * overlapping ones merged, then the closest pairs merged until at most [Fill.MAX_CROPS] remain.
     * Each side is grown to at least 512 px where the frame allows, so small people are
     * inpainted at native resolution rather than upscaled.
     */
    private fun crops(mask: Mask, fullW: Int, fullH: Int): List<IntArray> {
        val groups = mask.boxes.map { (cx, cy, bw, bh) ->
            val w = bw * fullW; val h = bh * fullH
            val margin = 0.3f * max(w, h) + 16f
            floatArrayOf(cx * fullW - w / 2 - margin, cy * fullH - h / 2 - margin, cx * fullW + w / 2 + margin, cy * fullH + h / 2 + margin)
        }.toMutableList()
        fun union(a: FloatArray, b: FloatArray) = floatArrayOf(min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
        fun overlaps(a: FloatArray, b: FloatArray) = a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]
        var merged = true
        while (merged) {
            merged = false
            loop@ for (i in groups.indices) for (j in i + 1 until groups.size) {
                if (overlaps(groups[i], groups[j])) {
                    groups[i] = union(groups[i], groups[j]); groups.removeAt(j); merged = true; break@loop
                }
            }
        }
        while (groups.size > Fill.MAX_CROPS) {
            var best = Pair(0, 1); var bestArea = Float.MAX_VALUE
            for (i in groups.indices) for (j in i + 1 until groups.size) {
                val u = union(groups[i], groups[j]); val area = (u[2] - u[0]) * (u[3] - u[1])
                if (area < bestArea) { bestArea = area; best = Pair(i, j) }
            }
            groups[best.first] = union(groups[best.first], groups[best.second]); groups.removeAt(best.second)
        }
        return groups.map { g ->
            val w = max(g[2] - g[0], min(SIZE, fullW).toFloat()).coerceAtMost(fullW.toFloat()).toInt()
            val h = max(g[3] - g[1], min(SIZE, fullH).toFloat()).coerceAtMost(fullH.toFloat()).toInt()
            val x = ((g[0] + g[2]) / 2 - w / 2f).toInt().coerceIn(0, fullW - w)
            val y = ((g[1] + g[3]) / 2 - h / 2f).toInt().coerceIn(0, fullH - h)
            intArrayOf(x, y, w, h)
        }
    }

    override fun close() {
        worker.execute {
            model?.close(); model = null
        }
        worker.shutdown()
    }

    companion object {
        private const val TAG = "LocalInpainter"
        const val MODEL_FILE = "lama_dilated_places.tflite"
        private const val SIZE = 512
        private const val ATLAS = 1024
        private const val HOLE_GROW = 4
        private const val INV_255 = 1f / 255f
    }
}
