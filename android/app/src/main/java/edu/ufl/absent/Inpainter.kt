package edu.ufl.absent

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.max
import kotlin.math.min

/**
 * Filled crops packed in one bitmap. Crop i covers rects[4i..4i+3] = (x, y, width, height) of the
 * upright camera image and atlasRects[4i..4i+3] of [bitmap], both normalised to 0–1.
 */
class Fill(val bitmap: Bitmap, val count: Int, val rects: FloatArray, val atlasRects: FloatArray) {
    companion object {
        const val MAX_CROPS = 4
        fun single(bitmap: Bitmap, rect: FloatArray) =
            Fill(bitmap, 1, rect.copyOf(4 * MAX_CROPS), floatArrayOf(0f, 0f, 1f, 1f).copyOf(4 * MAX_CROPS))
    }
}

/**
 * Sends a full-resolution crop around the people, with its mask, to the laptop's LaMa server
 * (server/inpaint.py, reached through `adb reverse tcp:8000 tcp:8000`) and keeps the latest fill.
 * One request is in flight at a time; frames offered meanwhile are dropped.
 */
class Inpainter(private val url: String = "http://127.0.0.1:8000/api/inpaint-frame") : FillSource {
    override val label = "laptop LaMa"
    override val roundTripMs = Samples()
    val serverMs = Samples()
    @Volatile override var completed = 0
        private set
    @Volatile override var lastError: String? = null
        private set
    @Volatile override var lastCrop = "—"
        private set
    private val worker = Executors.newSingleThreadExecutor()

    private val slot = java.util.concurrent.Semaphore(1)

    override fun submit(frame: YuvFrame, rotation: Int, mask: Mask, done: (Fill?) -> Unit) {
        slot.acquire()
        worker.execute {
            var fill: Fill? = null
            try {
                fill = send(frame, rotation, mask).also { lastError = null }
            } catch (e: Exception) {
                lastError = e.javaClass.simpleName
                Log.w(TAG, "inpaint request failed", e)
                Thread.sleep(500)
            } finally {
                slot.release()
                done(fill)
            }
        }
    }

    private fun send(frame: YuvFrame, rotation: Int, mask: Mask): Fill? {
        val start = System.nanoTime()
        val fullW = frame.uprightWidth(rotation)
        val fullH = frame.uprightHeight(rotation)
        val crop = cropAround(mask, fullW, fullH) ?: return null
        val (x0, y0, w, h) = crop
        lastCrop = "${w}×$h"

        val pixels = IntArray(w * h)
        for (y in 0 until h) for (x in 0 until w) pixels[y * w + x] = frame.argbUpright(x0 + x, y0 + y, rotation)
        val jpeg = ByteArrayOutputStream().also {
            Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888).compress(Bitmap.CompressFormat.JPEG, 92, it)
        }.toByteArray()

        // Mask at a quarter of the crop resolution, bilinearly sampled from the model mask.
        val mw = max(1, w / 4)
        val mh = max(1, h / 4)
        val maskBytes = ByteArray(mw * mh)
        for (my in 0 until mh) for (mx in 0 until mw) {
            val u = (x0 + (mx + 0.5f) * w / mw) / fullW * mask.width - 0.5f
            val v = (y0 + (my + 0.5f) * h / mh) / fullH * mask.height - 0.5f
            maskBytes[my * mw + mx] = sample(mask, u, v).toByte()
        }

        var fill: Fill? = null
        val connection = URL(url).openConnection() as HttpURLConnection
        try {
            connection.requestMethod = "POST"
            connection.doOutput = true
            connection.connectTimeout = 2000
            connection.readTimeout = 5000
            connection.setRequestProperty("Content-Type", "application/octet-stream")
            DataOutputStream(connection.outputStream).use { out ->
                out.writeInt(jpeg.size)
                out.write(jpeg)
                out.writeShort(mw)
                out.writeShort(mh)
                out.write(maskBytes)
            }
            check(connection.responseCode == 200) { "HTTP ${connection.responseCode}" }
            val body = connection.inputStream.use { it.readBytes() }
            val bitmap = BitmapFactory.decodeByteArray(body, 0, body.size)
            val rect = floatArrayOf(x0.toFloat() / fullW, y0.toFloat() / fullH, w.toFloat() / fullW, h.toFloat() / fullH)
            fill = Fill.single(bitmap, rect)
            connection.getHeaderField("X-Inference-Ms")?.toDoubleOrNull()?.let { serverMs.add(it) }
        } finally {
            connection.disconnect()
        }
        completed++
        roundTripMs.add((System.nanoTime() - start) / 1e6)
        return fill
    }

    /**
     * Upright pixel rectangle (x, y, w, h) around every masked pixel, with a margin for LaMa's
     * context, grown to at least [MIN_SIDE] per side where the frame allows (no upscaling of
     * small holes) and capped at [MAX_SIDE]. Null when nothing is masked.
     */
    private fun cropAround(mask: Mask, fullW: Int, fullH: Int): IntArray? {
        var minX = Int.MAX_VALUE; var minY = Int.MAX_VALUE; var maxX = -1; var maxY = -1
        for (y in 0 until mask.height) for (x in 0 until mask.width) {
            if ((mask.bytes[y * mask.width + x].toInt() and 0xff) > 127) {
                minX = min(minX, x); maxX = max(maxX, x); minY = min(minY, y); maxY = max(maxY, y)
            }
        }
        if (maxX < 0) return null
        val sx = fullW.toFloat() / mask.width
        val sy = fullH.toFloat() / mask.height
        var left = minX * sx; var right = (maxX + 1) * sx
        var top = minY * sy; var bottom = (maxY + 1) * sy
        val margin = 0.25f * max(right - left, bottom - top) + 24f
        left -= margin; right += margin; top -= margin; bottom += margin
        val w = (right - left).coerceIn(min(MIN_SIDE, fullW).toFloat(), min(MAX_SIDE, fullW).toFloat()).toInt()
        val h = (bottom - top).coerceIn(min(MIN_SIDE, fullH).toFloat(), min(MAX_SIDE, fullH).toFloat()).toInt()
        val x = ((left + right) / 2 - w / 2f).toInt().coerceIn(0, fullW - w)
        val y = ((top + bottom) / 2 - h / 2f).toInt().coerceIn(0, fullH - h)
        return intArrayOf(x, y, w, h)
    }

    private fun sample(mask: Mask, u: Float, v: Float): Int {
        val x = u.coerceIn(0f, mask.width - 1f)
        val y = v.coerceIn(0f, mask.height - 1f)
        val x0 = x.toInt(); val y0 = y.toInt()
        val x1 = min(x0 + 1, mask.width - 1); val y1 = min(y0 + 1, mask.height - 1)
        val fx = x - x0; val fy = y - y0
        fun at(px: Int, py: Int) = (mask.bytes[py * mask.width + px].toInt() and 0xff).toFloat()
        val top = at(x0, y0) * (1 - fx) + at(x1, y0) * fx
        val bottom = at(x0, y1) * (1 - fx) + at(x1, y1) * fx
        return (top * (1 - fy) + bottom * fy).toInt()
    }

    override fun close() { worker.shutdown() }

    companion object {
        private const val TAG = "Inpainter"
        private const val MIN_SIDE = 512
        private const val MAX_SIDE = 2048
    }
}
