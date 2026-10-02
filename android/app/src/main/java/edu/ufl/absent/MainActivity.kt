package edu.ufl.absent

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Color
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.opengl.GLES11Ext
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import com.google.ar.core.ArCoreApk
import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import java.util.EnumSet
import com.google.ar.core.Config
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.NotYetAvailableException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * Absent on the phone: ARCore camera (autofocus), YOLO person mask, LaMa fill. The mask and
 * removed views are synced: each shows a camera frame kept on the GPU together with the mask
 * and fill computed from that same frame, so it runs at processing speed, slightly delayed.
 * The camera view is live.
 */
class MainActivity : Activity(), GLSurfaceView.Renderer {
    private lateinit var glView: GLSurfaceView
    private lateinit var hud: TextView
    private lateinit var viewButton: Button
    private lateinit var focusButton: Button
    private lateinit var modelButton: Button
    private var modelIndex = 0
    private var session: Session? = null
    private var installRequested = false
    private var masker: PersonMasker? = null

    /** 0 live camera, 1 synced mask overlay, 2 synced removal (people replaced by that frame's fill). */
    @Volatile private var viewMode = 2
    /** Fill sources, cycled by the Fill button: phone NPU, phone GPU, laptop over USB. */
    private var fillIndex = 0
    private val inpainterLazy = lazy { makeFillSource(0) }
    private var inpainterOverride: FillSource? = null
    private var inpainter: FillSource
        get() = inpainterOverride ?: inpainterLazy.value
        set(value) { inpainterOverride = value }
    private lateinit var fillButton: Button

    private fun makeFillSource(index: Int): FillSource = when (index) {
        0 -> LocalInpainter(this, Backend.NPU)
        1 -> LocalInpainter(this, Backend.GPU)
        else -> Inpainter()
    }
    @Volatile private var autoFocus = true
    @Volatile private var sensorRotation = 90
    @Volatile private var viewportChanged = false
    private var viewportWidth = 0
    private var viewportHeight = 0

    // GL state
    private var cameraTexture = 0
    private var maskTexture = 0
    private var fillTexture = 0
    private val pending = java.util.concurrent.atomic.AtomicReference<Processed?>(null)
    private var shown: Processed? = null
    private var shownSlot = -1
    private var hasFill = false
    private val fillRects = FloatArray(4 * Fill.MAX_CROPS)
    private val atlasRects = FloatArray(4 * Fill.MAX_CROPS)
    private var fillCount = 0
    private var lastFillsCompleted = 0
    private var program = 0
    private var copyProgram = 0
    private var textureNameSet = false
    // Ring of camera frames (copied from the camera texture when the masker accepts a frame),
    // so a processed result can be shown over the exact frame it came from.
    private val ringTextures = IntArray(RING)
    private val ringFramebuffers = IntArray(RING)
    private val ringTimestamps = LongArray(RING)
    private var ringNext = 0
    private var ringWidth = 0
    private var ringHeight = 0
    private val shownFrameMs = Samples()
    private var lastShownNs = 0L
    private var latencyMs = 0.0
    private val ndcQuad = floats(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)
    private val cameraUv = floats(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
    private val imageUv = floats(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)

    // Stats (GL thread writes, UI thread reads)
    private val frameMs = Samples()
    private var lastFrameNs = 0L
    private var lastHudNs = 0L
    private var lastMasksCompleted = 0
    private var trackingState = "—"
    private var depthMode = "—"
    private var imageSize = "—"
    @Volatile private var frames = 0
    @Volatile private var framesWithoutTracking = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        glView = GLSurfaceView(this).apply {
            preserveEGLContextOnPause = true
            setEGLContextClientVersion(3)
            setEGLConfigChooser(8, 8, 8, 8, 16, 0)
            setRenderer(this@MainActivity)
            renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        }
        hud = TextView(this).apply {
            setTextColor(Color.WHITE)
            setBackgroundColor(0x99000000.toInt())
            textSize = 13f
            setPadding(24, 24, 24, 24)
            typeface = android.graphics.Typeface.MONOSPACE
        }
        viewButton = Button(this).apply {
            text = VIEW_LABELS[viewMode]
            setOnClickListener { viewMode = (viewMode + 1) % VIEW_LABELS.size; text = VIEW_LABELS[viewMode] }
        }
        focusButton = Button(this).apply {
            text = "Focus: auto"
            setOnClickListener { autoFocus = !autoFocus; text = if (autoFocus) "Focus: auto" else "Focus: fixed"; glView.queueEvent { configure() } }
        }
        modelButton = Button(this).apply {
            text = MODELS[modelIndex].label
            setOnClickListener {
                modelIndex = (modelIndex + 1) % MODELS.size
                text = MODELS[modelIndex].label
                masker?.close()
                masker = newMasker()
            }
        }
        fillButton = Button(this).apply {
            text = "Fill: ${inpainter.label}"
            setOnClickListener {
                fillIndex = (fillIndex + 1) % 3
                inpainter.close()
                inpainter = makeFillSource(fillIndex)
                masker?.inpainter = inpainter
                text = "Fill: ${inpainter.label}"
            }
        }
        val buttons = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(LinearLayout(this@MainActivity).apply { addView(viewButton); addView(focusButton) })
            addView(LinearLayout(this@MainActivity).apply { addView(modelButton); addView(fillButton) })
        }
        setContentView(FrameLayout(this).apply {
            addView(glView)
            addView(hud, FrameLayout.LayoutParams(-1, -2, Gravity.TOP).apply { topMargin = 80 })
            addView(buttons, FrameLayout.LayoutParams(-2, -2, Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL).apply { bottomMargin = 120 })
        })
    }

    override fun onResume() {
        super.onResume()
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), 0)
            return
        }
        if (session == null) {
            try {
                if (ArCoreApk.getInstance().requestInstall(this, !installRequested) == ArCoreApk.InstallStatus.INSTALL_REQUESTED) {
                    installRequested = true
                    return
                }
                session = Session(this)
                chooseCameraConfig(session!!)
                configure()
                sensorRotation = sensorOrientation(session!!.cameraConfig.cameraId)
            } catch (e: Exception) {
                Log.e(TAG, "ARCore session failed", e)
                Toast.makeText(this, "ARCore failed: ${e.message}", Toast.LENGTH_LONG).show()
                return
            }
        }
        if (masker == null) masker = newMasker()
        try {
            session!!.resume()
        } catch (e: Exception) {
            Log.e(TAG, "camera unavailable", e)
            Toast.makeText(this, "Camera unavailable: ${e.message}", Toast.LENGTH_LONG).show()
            session = null
            return
        }
        glView.onResume()
    }

    override fun onPause() {
        super.onPause()
        glView.onPause()
        session?.pause()
    }

    override fun onDestroy() {
        masker?.close()
        inpainter.close()
        session?.close()
        session = null
        super.onDestroy()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, results: IntArray) {
        if (results.firstOrNull() != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    private fun newMasker() = PersonMasker(this, MODELS[modelIndex]).also {
        it.inpainter = inpainter
        it.sink = { result -> pending.set(result) }
    }

    private fun configure() {
        val s = session ?: return
        val config = Config(s).apply {
            focusMode = if (autoFocus) Config.FocusMode.AUTO else Config.FocusMode.FIXED
            updateMode = Config.UpdateMode.BLOCKING
            depthMode = if (s.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) Config.DepthMode.AUTOMATIC else Config.DepthMode.DISABLED
        }
        s.configure(config)
        depthMode = config.depthMode.name
    }

    /** Highest CPU-image resolution at 30 fps, so crops sent for inpainting keep their detail. */
    private fun chooseCameraConfig(s: Session) {
        val filter = CameraConfigFilter(s).setTargetFps(EnumSet.of(CameraConfig.TargetFps.TARGET_FPS_30))
        val configs = s.getSupportedCameraConfigs(filter)
        configs.forEach { Log.i(TAG, "camera config: cpu ${it.imageSize} gpu ${it.textureSize} depth ${it.depthSensorUsage}") }
        configs.maxByOrNull { it.imageSize.width * it.imageSize.height }?.let { s.cameraConfig = it }
    }

    private fun sensorOrientation(cameraId: String): Int =
        getSystemService(CameraManager::class.java).getCameraCharacteristics(cameraId)
            .get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90

    // --- Renderer ---

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES30.glClearColor(0f, 0f, 0f, 1f)
        val textures = IntArray(3)
        GLES30.glGenTextures(3, textures, 0)
        cameraTexture = textures[0]
        maskTexture = textures[1]
        fillTexture = textures[2]
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, fillTexture)
        setLinearClamp(GLES30.GL_TEXTURE_2D)
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTexture)
        setLinearClamp(GLES11Ext.GL_TEXTURE_EXTERNAL_OES)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, maskTexture)
        setLinearClamp(GLES30.GL_TEXTURE_2D)
        program = linkProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        copyProgram = linkProgram(COPY_VERTEX_SHADER, COPY_FRAGMENT_SHADER)
        textureNameSet = false
        ringWidth = 0
        shown = null
        shownSlot = -1
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES30.glViewport(0, 0, width, height)
        viewportWidth = width
        viewportHeight = height
        viewportChanged = true
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        val s = session ?: return
        if (!textureNameSet) { s.setCameraTextureName(cameraTexture); textureNameSet = true }
        if (viewportChanged) {
            s.setDisplayGeometry(display!!.rotation, viewportWidth, viewportHeight)
            viewportChanged = false
        }
        val frame = try { s.update() } catch (e: Exception) { Log.w(TAG, "update failed", e); return }

        val now = System.nanoTime()
        if (lastFrameNs != 0L) frameMs.add((now - lastFrameNs) / 1e6)
        lastFrameNs = now
        frames++

        if (frame.hasDisplayGeometryChanged() || frame.timestamp == 0L) {
            frame.transformCoordinates2d(Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES, ndcQuad, Coordinates2d.TEXTURE_NORMALIZED, cameraUv)
            frame.transformCoordinates2d(Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES, ndcQuad, Coordinates2d.IMAGE_NORMALIZED, imageUv)
        }
        if (frame.timestamp == 0L) return

        val camera = frame.camera
        trackingState = camera.trackingState.name
        if (camera.trackingState != TrackingState.TRACKING) framesWithoutTracking++

        masker?.let { m ->
            try {
                frame.acquireCameraImage().use { image ->
                    imageSize = "${image.width}×${image.height}"
                    if (m.offer(image, sensorRotation, frame.timestamp)) keepFrame(frame.timestamp, camera)
                }
            } catch (_: NotYetAvailableException) {
            }
        }
        showNewest(frame.timestamp)

        draw()
        if (now - lastHudNs > 500_000_000L) { lastHudNs = now; postHud() }
    }

    /** Copies the current camera texture into the next ring slot, tagged with its timestamp. */
    private fun keepFrame(timestamp: Long, camera: com.google.ar.core.Camera) {
        if (ringWidth == 0) {
            val size = camera.textureIntrinsics.imageDimensions
            ringWidth = size[0]; ringHeight = size[1]
            GLES30.glGenTextures(RING, ringTextures, 0)
            GLES30.glGenFramebuffers(RING, ringFramebuffers, 0)
            for (i in 0 until RING) {
                GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, ringTextures[i])
                setLinearClamp(GLES30.GL_TEXTURE_2D)
                GLES30.glTexStorage2D(GLES30.GL_TEXTURE_2D, 1, GLES30.GL_RGBA8, ringWidth, ringHeight)
                GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, ringFramebuffers[i])
                GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0, GLES30.GL_TEXTURE_2D, ringTextures[i], 0)
                ringTimestamps[i] = 0L
            }
        }
        // Never overwrite the slot on screen.
        if (ringNext == shownSlot) ringNext = (ringNext + 1) % RING
        val slot = ringNext
        ringNext = (ringNext + 1) % RING
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, ringFramebuffers[slot])
        GLES30.glViewport(0, 0, ringWidth, ringHeight)
        GLES30.glUseProgram(copyProgram)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTexture)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(copyProgram, "uCamera"), 0)
        attribute(0, ndcQuad)
        GLES30.glDisableVertexAttribArray(1)
        GLES30.glDisableVertexAttribArray(2)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        GLES30.glViewport(0, 0, viewportWidth, viewportHeight)
        ringTimestamps[slot] = timestamp
    }

    /** Takes the newest processed frame whose camera frame is still in the ring and uploads its mask and fill. */
    private fun showNewest(now: Long) {
        val result = pending.getAndSet(null) ?: return
        if (shown != null && result.timestamp <= shown!!.timestamp) return
        val slot = ringTimestamps.indexOfFirst { it == result.timestamp }
        if (slot < 0) return
        shown = result
        shownSlot = slot
        uploadMask(result.mask)
        val fill = result.fill
        hasFill = fill != null
        if (fill != null) {
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, fillTexture)
            android.opengl.GLUtils.texImage2D(GLES30.GL_TEXTURE_2D, 0, fill.bitmap, 0)
            fill.rects.copyInto(fillRects)
            fill.atlasRects.copyInto(atlasRects)
            fillCount = fill.count
        } else {
            fillCount = 0
        }
        val t = System.nanoTime()
        if (lastShownNs != 0L) shownFrameMs.add((t - lastShownNs) / 1e6)
        lastShownNs = t
        latencyMs = (now - result.timestamp) / 1e6
    }

    private fun draw() {
        val synced = viewMode != 0 && shownSlot >= 0
        GLES30.glUseProgram(program)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTexture)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE3)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, if (synced) ringTextures[shownSlot] else 0)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "uFrame"), 3)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "uSynced"), if (synced) 1 else 0)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, maskTexture)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "uCamera"), 0)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "uMask"), 1)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "uRot"), sensorRotation)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE2)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, fillTexture)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "uFill"), 2)
        GLES30.glUniform4fv(GLES30.glGetUniformLocation(program, "uFillRect"), Fill.MAX_CROPS, fillRects, 0)
        GLES30.glUniform4fv(GLES30.glGetUniformLocation(program, "uAtlasRect"), Fill.MAX_CROPS, atlasRects, 0)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "uFillCount"), fillCount)
        val mode = if (synced) viewMode else 0
        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "uMode"), mode)
        attribute(0, ndcQuad)
        attribute(1, cameraUv)
        attribute(2, imageUv)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
    }

    private fun uploadMask(mask: Mask) {
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, maskTexture)
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, 1)
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_R8, mask.width, mask.height, 0,
            GLES30.GL_RED, GLES30.GL_UNSIGNED_BYTE, ByteBuffer.wrap(mask.bytes))
    }

    private fun postHud() {
        val m = masker
        val fps = frameMs.median()?.let { 1000.0 / it }
        val completed = m?.masksCompleted ?: 0
        val maskFps = (completed - lastMasksCompleted) * 2
        lastMasksCompleted = completed
        val text = buildString {
            appendLine("camera  ${fmt(fps, 1)} fps   image $imageSize   rot $sensorRotation")
            appendLine("shown   ${fmt(shownFrameMs.median()?.let { 1000.0 / it }, 1)} fps synced   delay ${fmt(latencyMs, 0)} ms   people ${shown?.mask?.people ?: 0}")
            appendLine("mask    ${m?.spec?.label ?: "—"} (${m?.delegate ?: "—"})   ~$maskFps fps")
            appendLine("  ms    total ${fmt(m?.totalMs?.median(), 1)} = yuv ${fmt(m?.convertMs?.median(), 1)} + model ${fmt(m?.inferMs?.median(), 1)} + decode ${fmt(m?.decodeMs?.median(), 1)}")
            appendLine("track   $trackingState   lost ${framesWithoutTracking}/$frames frames")
            appendLine("focus   ${if (autoFocus) "AUTO" else "FIXED"}   depth $depthMode")
            val source = inpainter
            val fills = source.completed
            appendLine("fill    ${source.label}   ~${(fills - lastFillsCompleted) * 2} fps   total ${fmt(source.roundTripMs.median(), 0)} ms")
            append(when (source) {
                is LocalInpainter -> "  lama  ${source.loadedAs}   model ${fmt(source.modelMs.median(), 1)} ms/crop   prep ${fmt(source.prepMs.median(), 0)} ms   crops ${source.lastCrop}"
                is Inpainter -> "  lama  server ${fmt(source.serverMs.median(), 0)} ms   crop ${source.lastCrop}"
                else -> ""
            })
            source.lastError?.let { append("\n  ERR $it") }
            lastFillsCompleted = fills
        }
        runOnUiThread { hud.text = text }
    }

    // --- GL helpers ---

    private fun attribute(index: Int, data: FloatBuffer) {
        data.position(0)
        GLES30.glEnableVertexAttribArray(index)
        GLES30.glVertexAttribPointer(index, 2, GLES30.GL_FLOAT, false, 0, data)
    }

    private fun setLinearClamp(target: Int) {
        GLES30.glTexParameteri(target, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(target, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(target, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(target, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
    }

    private fun linkProgram(vertex: String, fragment: String): Int {
        val p = GLES30.glCreateProgram()
        GLES30.glAttachShader(p, compile(GLES30.GL_VERTEX_SHADER, vertex))
        GLES30.glAttachShader(p, compile(GLES30.GL_FRAGMENT_SHADER, fragment))
        GLES30.glLinkProgram(p)
        val status = IntArray(1)
        GLES30.glGetProgramiv(p, GLES30.GL_LINK_STATUS, status, 0)
        check(status[0] != 0) { "link failed: ${GLES30.glGetProgramInfoLog(p)}" }
        return p
    }

    private fun compile(type: Int, source: String): Int {
        val shader = GLES30.glCreateShader(type)
        GLES30.glShaderSource(shader, source)
        GLES30.glCompileShader(shader)
        val status = IntArray(1)
        GLES30.glGetShaderiv(shader, GLES30.GL_COMPILE_STATUS, status, 0)
        check(status[0] != 0) { "compile failed: ${GLES30.glGetShaderInfoLog(shader)}" }
        return shader
    }

    companion object {
        private const val TAG = "Absent"

        private val VIEW_LABELS = listOf("View: live camera", "View: mask", "View: removed")
        private const val RING = 6

        private const val COPY_VERTEX_SHADER = """#version 300 es
layout(location = 0) in vec2 aPosition;
out vec2 vUv;
void main() {
  gl_Position = vec4(aPosition, 0.0, 1.0);
  vUv = aPosition * 0.5 + 0.5;
}
"""

        // Identity copy in texture space, so a ring frame is sampled with the same coordinates as the camera texture.
        private const val COPY_FRAGMENT_SHADER = """#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision mediump float;
uniform samplerExternalOES uCamera;
in vec2 vUv;
out vec4 outColor;
void main() { outColor = texture(uCamera, vUv); }
"""

        private fun floats(vararg values: Float): FloatBuffer =
            ByteBuffer.allocateDirect(values.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().apply { put(values); position(0) }

        private fun fmt(value: Double?, digits: Int): String = value?.let { "%.${digits}f".format(it) } ?: "—"

        private const val VERTEX_SHADER = """#version 300 es
layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec2 aCameraUv;
layout(location = 2) in vec2 aImageUv;
out vec2 vCameraUv;
out vec2 vImageUv;
void main() {
  gl_Position = vec4(aPosition, 0.0, 1.0);
  vCameraUv = aCameraUv;
  vImageUv = aImageUv;
}
"""

        // The mask was computed on the camera image rotated clockwise by uRot, so image
        // coordinates are rotated the same way before sampling it.
        private const val FRAGMENT_SHADER = """#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision mediump float;
uniform samplerExternalOES uCamera;
uniform sampler2D uFrame;
uniform int uSynced;
uniform sampler2D uMask;
uniform int uRot;
uniform sampler2D uFill;
uniform vec4 uFillRect[4];
uniform vec4 uAtlasRect[4];
uniform int uFillCount;
uniform int uMode;
in vec2 vCameraUv;
in vec2 vImageUv;
out vec4 outColor;
vec2 upright(vec2 p) {
  if (uRot == 90) return vec2(1.0 - p.y, p.x);
  if (uRot == 180) return vec2(1.0 - p.x, 1.0 - p.y);
  if (uRot == 270) return vec2(p.y, 1.0 - p.x);
  return p;
}
void main() {
  vec4 color = uSynced == 1 ? texture(uFrame, vCameraUv) : texture(uCamera, vCameraUv);
  vec2 uv = upright(vImageUv);
  if (uMode == 1) {
    float person = texture(uMask, uv).r;
    color.rgb = mix(color.rgb, vec3(1.0, 0.1, 0.1), 0.6 * person);
  } else if (uMode == 2) {
    // Later crops win where crops overlap.
    float inside = 0.0;
    vec3 fill = color.rgb;
    for (int i = 0; i < 4; i++) {
      if (i >= uFillCount) break;
      vec2 f = (uv - uFillRect[i].xy) / uFillRect[i].zw;
      if (f.x >= 0.0 && f.x <= 1.0 && f.y >= 0.0 && f.y <= 1.0) {
        fill = texture(uFill, uAtlasRect[i].xy + f * uAtlasRect[i].zw).rgb;
        inside = 1.0;
      }
    }
    float person = smoothstep(0.1, 0.6, texture(uMask, uv).r);
    // A person without a fill (fill failed or still loading) is tinted, so it is never shown as removed.
    color.rgb = mix(color.rgb, mix(vec3(1.0, 0.1, 0.1), fill, inside), person);
  }
  outColor = color;
}
"""
    }
}
