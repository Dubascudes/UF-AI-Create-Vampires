package edu.ufl.absent

import android.media.Image
import java.nio.ByteBuffer
import kotlin.math.max

/** Reusable copy of a YUV_420_888 camera image, so the ARCore image can be closed at once. */
class YuvFrame {
    var y = ByteArray(0); var u = ByteArray(0); var v = ByteArray(0)
    var width = 0; var height = 0; var yRow = 0; var uvRow = 0; var uvPixel = 0

    fun copy(image: Image) {
        width = image.width; height = image.height
        val planes = image.planes
        yRow = planes[0].rowStride
        uvRow = planes[1].rowStride
        uvPixel = planes[1].pixelStride
        y = fill(y, planes[0].buffer); u = fill(u, planes[1].buffer); v = fill(v, planes[2].buffer)
    }

    /** An independent copy, for work that outlives the next camera frame. */
    fun snapshot(): YuvFrame = YuvFrame().also {
        it.y = y.copyOf(); it.u = u.copyOf(); it.v = v.copyOf()
        it.width = width; it.height = height; it.yRow = yRow; it.uvRow = uvRow; it.uvPixel = uvPixel
    }

    fun uprightWidth(rotation: Int): Int = if (rotation == 90 || rotation == 270) height else width
    fun uprightHeight(rotation: Int): Int = if (rotation == 90 || rotation == 270) width else height

    /** Opaque ARGB at upright pixel ([ux], [uy]) of the image rotated clockwise by [rotation]. */
    fun argbUpright(ux: Int, uy: Int, rotation: Int): Int {
        val sx: Int
        val sy: Int
        when (rotation) {
            90 -> { sx = uy; sy = height - 1 - ux }
            180 -> { sx = width - 1 - ux; sy = height - 1 - uy }
            270 -> { sx = width - 1 - uy; sy = ux }
            else -> { sx = ux; sy = uy }
        }
        val c = 1192 * max((y[sy * yRow + sx].toInt() and 0xff) - 16, 0)
        val uvIndex = (sy shr 1) * uvRow + (sx shr 1) * uvPixel
        val cu = (u[uvIndex].toInt() and 0xff) - 128
        val cv = (v[uvIndex].toInt() and 0xff) - 128
        val r = ((c + 1634 * cv) shr 10).coerceIn(0, 255)
        val g = ((c - 833 * cv - 400 * cu) shr 10).coerceIn(0, 255)
        val b = ((c + 2066 * cu) shr 10).coerceIn(0, 255)
        return (0xff shl 24) or (r shl 16) or (g shl 8) or b
    }

    private fun fill(target: ByteArray, buffer: ByteBuffer): ByteArray {
        val out = if (target.size == buffer.remaining()) target else ByteArray(buffer.remaining())
        buffer.get(out)
        buffer.rewind()
        return out
    }
}
