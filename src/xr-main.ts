import { colorizeDepth, depthStats, depthToMeters, type DepthFormat } from './xr-depth';
import { FrontCameraCheck } from './xr-front';
import { COPY_FRAGMENT, OVERLAY_FRAGMENT, compile, dataTexture } from './xr-gl';
import { FrameRecorder, PersonMasker, READ_WIDTH, Samples, fmt, saveReport } from './xr-shared';
import './xr.css';

// Week-1 feasibility check for phone AR: which WebXR features the phone grants, how fast camera
// readback and person segmentation run, what the depth looks like, and short recordings of
// camera + pose + depth that land in recordings/<id>/ on the dev machine. The front camera has
// no WebXR support, so it gets a separate getUserMedia check (xr-front.ts).

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <main class="xr-landing" id="landing">
    <p class="xr-eyebrow">ABSENT / PHONE AR CHECK</p>
    <h1>Can this phone<br />see the room?</h1>
    <p>Tests camera access, tracking, depth, and person masks. Recordings save to the laptop.</p>
    <ul class="xr-checks" id="checks"></ul>
    <button id="startButton" class="xr-button" type="button" disabled>Start AR check · rear camera</button>
    <button id="frontButton" class="xr-button xr-button-secondary" type="button" disabled>Start front camera check</button>
    <p class="xr-note">The front camera runs without AR tracking or depth; ARCore provides both only for the rear camera.</p>
    <pre class="xr-report" id="report" hidden></pre>
  </main>
  <div class="xr-front-stage" id="frontStage" hidden></div>
  <div class="xr-overlay" id="overlay" hidden>
    <pre class="xr-hud" id="hud">Starting…</pre>
    <div class="xr-bar">
      <button id="viewButton" class="xr-chip" type="button">View: camera</button>
      <button id="segmentButton" class="xr-chip" type="button">Mask: off</button>
      <button id="recordButton" class="xr-chip" type="button">● Record</button>
      <button id="exitButton" class="xr-chip" type="button">Exit</button>
    </div>
  </div>`;

const select = <T extends Element>(query: string): T => document.querySelector<T>(query)!;
const landing = select<HTMLElement>('#landing');
const checks = select<HTMLUListElement>('#checks');
const startButton = select<HTMLButtonElement>('#startButton');
const frontButton = select<HTMLButtonElement>('#frontButton');
const report = select<HTMLPreElement>('#report');
const frontStage = select<HTMLDivElement>('#frontStage');
const overlay = select<HTMLDivElement>('#overlay');
const hud = select<HTMLPreElement>('#hud');
const viewButton = select<HTMLButtonElement>('#viewButton');
const segmentButton = select<HTMLButtonElement>('#segmentButton');
const recordButton = select<HTMLButtonElement>('#recordButton');
const exitButton = select<HTMLButtonElement>('#exitButton');

const VIEWS = ['camera', 'depth', 'mask'] as const;
type ViewMode = typeof VIEWS[number];

function addCheck(label: string, ok: boolean, detail = ''): void {
  const item = document.createElement('li');
  item.className = ok ? 'ok' : 'bad';
  item.textContent = `${ok ? '✓' : '✗'} ${label}${detail ? ` — ${detail}` : ''}`;
  checks.append(item);
}

async function checkSupport(): Promise<void> {
  addCheck('Secure context', window.isSecureContext, window.isSecureContext ? '' : 'open via localhost (adb reverse) or HTTPS');
  const supported = !!navigator.xr && await navigator.xr.isSessionSupported('immersive-ar').catch(() => false);
  addCheck('WebXR immersive-ar', supported, supported ? '' : 'use Chrome on an ARCore phone');
  const cameras = await navigator.mediaDevices?.enumerateDevices().catch(() => []) ?? [];
  const hasCamera = cameras.some(device => device.kind === 'videoinput');
  addCheck('Camera for front check', hasCamera);
  startButton.disabled = !supported;
  frontButton.disabled = !hasCamera;
}

// ---------- Rear camera: WebXR session ----------

interface Run {
  session: XRSession;
  gl: WebGL2RenderingContext;
  space: XRReferenceSpace;
  binding: XRWebGLBinding | null;
  copyProgram: WebGLProgram;
  overlayProgram: WebGLProgram;
  vao: WebGLVertexArrayObject;
  readTexture: WebGLTexture;
  readFramebuffer: WebGLFramebuffer;
  readSize: { width: number; height: number } | null;
  readPixels: Uint8Array;
  readCanvas: HTMLCanvasElement;
  readContext: CanvasRenderingContext2D;
  depthTexture: WebGLTexture;
  maskTexture: WebGLTexture;
  maskReady: boolean;
  view: ViewMode;
  masker: PersonMasker;
  recorder: FrameRecorder;
  lastFrameAt: number;
  lastHudAt: number;
  frames: number;
  framesWithoutPose: number;
  framesWithCamera: number;
  framesWithDepth: number;
  maxPlanes: number;
  camera: { width: number; height: number } | null;
  depth: { width: number; height: number; format: string } | null;
  depthValid: Samples;
  depthMedian: Samples;
  frameMs: Samples;
  readMs: Samples;
  lastDepthStats: ReturnType<typeof depthStats> | null;
}

let run: Run | null = null;
let front: FrontCameraCheck | null = null;

async function startSession(): Promise<void> {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2', { xrCompatible: true, antialias: false })!;
  const session = await navigator.xr!.requestSession('immersive-ar', {
    requiredFeatures: ['local'],
    optionalFeatures: ['camera-access', 'depth-sensing', 'dom-overlay', 'plane-detection'],
    depthSensing: { usagePreference: ['cpu-optimized'], dataFormatPreference: ['luminance-alpha', 'float32', 'unsigned-short'] },
    domOverlay: { root: overlay },
  });
  showSessionUi();
  await session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });
  const space = await session.requestReferenceSpace('local');
  const features = session.enabledFeatures;
  const cameraGranted = !features || features.includes('camera-access');
  const readCanvas = document.createElement('canvas');
  run = {
    session, gl, space,
    binding: cameraGranted ? new XRWebGLBinding(session, gl) : null,
    copyProgram: compile(gl, COPY_FRAGMENT),
    overlayProgram: compile(gl, OVERLAY_FRAGMENT),
    vao: gl.createVertexArray()!,
    readTexture: dataTexture(gl), readFramebuffer: gl.createFramebuffer()!, readSize: null, readPixels: new Uint8Array(0),
    readCanvas, readContext: readCanvas.getContext('2d', { willReadFrequently: true })!,
    depthTexture: dataTexture(gl), maskTexture: dataTexture(gl), maskReady: false,
    view: 'camera', masker: new PersonMasker(), recorder: new FrameRecorder(),
    lastFrameAt: 0, lastHudAt: 0, frames: 0, framesWithoutPose: 0, framesWithCamera: 0, framesWithDepth: 0, maxPlanes: 0,
    camera: null, depth: null, depthValid: new Samples(), depthMedian: new Samples(),
    frameMs: new Samples(), readMs: new Samples(), lastDepthStats: null,
  };
  session.addEventListener('end', () => finishSession());
  session.requestAnimationFrame(onFrame);
}

/** Draw the camera texture into a small framebuffer and read it back top-down into readCanvas. */
function readCamera(current: Run, cameraTexture: WebGLTexture, camera: XRCamera): void {
  const { gl } = current;
  const width = READ_WIDTH;
  const height = Math.round(READ_WIDTH * camera.height / camera.width);
  if (!current.readSize || current.readSize.width !== width || current.readSize.height !== height) {
    gl.bindTexture(gl.TEXTURE_2D, current.readTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, current.readFramebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, current.readTexture, 0);
    current.readSize = { width, height };
    current.readPixels = new Uint8Array(width * height * 4);
    current.readCanvas.width = width;
    current.readCanvas.height = height;
  }
  const start = performance.now();
  gl.bindFramebuffer(gl.FRAMEBUFFER, current.readFramebuffer);
  gl.viewport(0, 0, width, height);
  gl.useProgram(current.copyProgram);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, cameraTexture);
  gl.uniform1i(gl.getUniformLocation(current.copyProgram, 'uTexture'), 0);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, current.readPixels);
  const image = current.readContext.createImageData(width, height);
  const rowBytes = width * 4;
  for (let y = 0; y < height; y++) {
    image.data.set(current.readPixels.subarray((height - 1 - y) * rowBytes, (height - y) * rowBytes), y * rowBytes);
  }
  current.readContext.putImageData(image, 0, 0);
  current.readMs.add(performance.now() - start);
}

function segment(current: Run, time: number): void {
  const mask = current.masker.segment(current.readCanvas, time);
  if (!mask) return;
  const { gl } = current;
  gl.bindTexture(gl.TEXTURE_2D, current.maskTexture);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, mask.width, mask.height, 0, gl.RED, gl.UNSIGNED_BYTE, mask.bytes);
  current.maskReady = true;
}

function record(current: Run, time: number, view: XRView, depth: XRCPUDepthInformation | null, depthFormat: string): void {
  // The depth buffer is only valid inside this animation frame, so it is copied here.
  current.recorder.capture(time, current.readCanvas, {
    source: 'rear-xr',
    cameraToWorld: Array.from(view.transform.matrix),
    projection: Array.from(view.projectionMatrix),
    depth: depth ? {
      width: depth.width, height: depth.height, format: depthFormat, rawValueToMeters: depth.rawValueToMeters,
      normDepthBufferFromNormView: Array.from(depth.normDepthBufferFromNormView.matrix),
    } : null,
  }, depth ? new Uint8Array(depth.data.slice(0)) : undefined);
}

function onFrame(time: number, frame: XRFrame): void {
  const current = run;
  if (!current || frame.session !== current.session) return;
  current.session.requestAnimationFrame(onFrame);
  const { gl, session } = current;
  if (current.lastFrameAt) current.frameMs.add(time - current.lastFrameAt);
  current.lastFrameAt = time;
  current.frames++;
  current.maxPlanes = Math.max(current.maxPlanes, frame.detectedPlanes?.size ?? 0);

  const layer = session.renderState.baseLayer!;
  const pose = frame.getViewerPose(current.space);
  gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  if (!pose) { current.framesWithoutPose++; updateHud(current, time, pose); return; }
  const view = pose.views[0];
  gl.bindVertexArray(current.vao);

  const depth = safeDepth(frame, view);
  const depthFormat = (session.depthDataFormat ?? 'luminance-alpha') as DepthFormat;
  const cameraTexture = current.binding && view.camera ? current.binding.getCameraImage(view.camera) : null;
  if (cameraTexture && view.camera) {
    current.framesWithCamera++;
    current.camera = { width: view.camera.width, height: view.camera.height };
    const recordDue = current.recorder.due(time);
    if (current.masker.segmenter || recordDue) {
      readCamera(current, cameraTexture, view.camera);
      segment(current, time);
    }
    if (recordDue) record(current, time, view, depth, depthFormat);
  }

  let meters: Float32Array | null = null;
  if (depth) {
    current.framesWithDepth++;
    current.depth = { width: depth.width, height: depth.height, format: depthFormat };
    if (current.view === 'depth' || time - current.lastHudAt >= 500) {
      meters = depthToMeters(depth.data, depthFormat, depth.rawValueToMeters, depth.width, depth.height);
    }
  }
  if (meters && time - current.lastHudAt >= 500) {
    current.lastDepthStats = depthStats(meters);
    current.depthValid.add(current.lastDepthStats.validFraction);
    if (current.lastDepthStats.validFraction) current.depthMedian.add(current.lastDepthStats.median);
  }

  const viewport = layer.getViewport(view)!;
  gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
  gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
  if (current.view === 'depth' && depth && meters) {
    gl.bindTexture(gl.TEXTURE_2D, current.depthTexture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, depth.width, depth.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, colorizeDepth(meters));
    drawOverlay(current, 1, current.depthTexture, depth.normDepthBufferFromNormView.matrix);
  } else if (current.view === 'mask' && current.maskReady && current.masker.segmenter) {
    drawOverlay(current, 2, current.maskTexture);
  }
  updateHud(current, time, pose);
}

function safeDepth(frame: XRFrame, view: XRView): XRCPUDepthInformation | null {
  try { return frame.getDepthInformation(view) ?? null; } catch { return null; }
}

function drawOverlay(current: Run, mode: number, texture: WebGLTexture, depthUv?: Float32Array): void {
  const { gl, overlayProgram } = current;
  gl.useProgram(overlayProgram);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.uniform1i(gl.getUniformLocation(overlayProgram, 'uTexture'), 0);
  gl.uniform1i(gl.getUniformLocation(overlayProgram, 'uMode'), mode);
  gl.uniformMatrix4fv(gl.getUniformLocation(overlayProgram, 'uDepthUv'), false, depthUv ?? new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]));
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

function updateHud(current: Run, time: number, pose: XRViewerPose | undefined): void {
  if (time - current.lastHudAt < 500) return;
  current.lastHudAt = time;
  const frameMs = current.frameMs.median();
  const stats = current.lastDepthStats;
  const features = current.session.enabledFeatures;
  hud.textContent = [
    `FPS ${fmt(frameMs && 1000 / frameMs, 1)}  ·  tracking ${pose ? (pose.emulatedPosition ? 'orientation only' : '6DoF') : 'LOST'}`,
    `camera ${current.camera ? `${current.camera.width}×${current.camera.height}` : current.binding ? 'no image yet' : 'NOT GRANTED'}  ·  readback ${fmt(current.readMs.median(), 1, ' ms')}`,
    `depth ${current.depth ? `${current.depth.width}×${current.depth.height} ${current.depth.format}` : 'none'}  ·  valid ${stats ? fmt(stats.validFraction * 100, 0, '%') : '—'}  ·  ${stats ? `${fmt(stats.min, 2)}–${fmt(stats.max, 2)} m, median ${fmt(stats.median, 2)} m` : ''}`,
    `mask ${current.masker.label}  ·  planes ${features && !features.includes('plane-detection') ? 'not granted' : `max ${current.maxPlanes}`}`,
    current.recorder.active ? current.recorder.label : `features: ${(features ?? ['(not reported)']).join(', ')}`,
  ].join('\n');
}

function finishSession(): void {
  const current = run;
  run = null;
  if (!current) return;
  current.masker.close();
  const frameMs = current.frameMs.median();
  showLanding({
    mode: 'rear-xr',
    finishedAt: new Date().toISOString(),
    userAgent: navigator.userAgent,
    enabledFeatures: current.session.enabledFeatures ?? null,
    depthUsage: current.session.depthUsage ?? null,
    depthDataFormat: current.session.depthDataFormat ?? null,
    frames: current.frames,
    medianFps: frameMs ? Number((1000 / frameMs).toFixed(1)) : null,
    framesWithoutPose: current.framesWithoutPose,
    framesWithCamera: current.framesWithCamera,
    framesWithDepth: current.framesWithDepth,
    camera: current.camera,
    readbackSize: current.readSize,
    medianReadbackMs: current.readMs.median(),
    segmenter: current.masker.delegate,
    medianSegmentMs: current.masker.ms.median(),
    depth: current.depth,
    medianDepthValidFraction: current.depthValid.median(),
    medianSceneDepthM: current.depthMedian.median(),
    maxPlanes: current.maxPlanes,
    recording: current.recorder.summary,
  });
}

// ---------- Shared session UI ----------

function showSessionUi(): void {
  report.hidden = true;
  landing.hidden = true;
  overlay.hidden = false;
  viewButton.textContent = front ? 'View: mask' : 'View: camera';
  segmentButton.textContent = 'Mask: off';
  recordButton.textContent = '● Record';
}

function showLanding(summary: object): void {
  report.textContent = JSON.stringify(summary, null, 2);
  report.hidden = false;
  overlay.hidden = true;
  landing.hidden = false;
  startButton.disabled = frontButton.disabled = false;
  saveReport(summary);
}

function startFailed(error: unknown): void {
  overlay.hidden = true;
  landing.hidden = false;
  startButton.disabled = frontButton.disabled = false;
  report.textContent = `Could not start: ${String(error)}`;
  report.hidden = false;
}

startButton.addEventListener('click', () => {
  startButton.disabled = frontButton.disabled = true;
  startSession().catch(startFailed);
});
frontButton.addEventListener('click', () => {
  startButton.disabled = frontButton.disabled = true;
  FrontCameraCheck.start(frontStage, hud).then(check => { front = check; showSessionUi(); }, startFailed);
});
viewButton.addEventListener('click', () => {
  if (front) { viewButton.textContent = front.cycleView(); return; }
  if (!run) return;
  run.view = VIEWS[(VIEWS.indexOf(run.view) + 1) % VIEWS.length];
  viewButton.textContent = `View: ${run.view}`;
});
segmentButton.addEventListener('click', async () => {
  segmentButton.disabled = true;
  try {
    if (front) {
      segmentButton.textContent = 'Mask: loading';
      segmentButton.textContent = await front.toggleMask();
    } else if (run) {
      const { masker } = run;
      if (masker.segmenter) { masker.close(); segmentButton.textContent = 'Mask: off'; return; }
      segmentButton.textContent = 'Mask: loading';
      await masker.load();
      if (run?.masker !== masker) { masker.close(); return; }
      segmentButton.textContent = masker.segmenter ? `Mask: ${masker.delegate}` : 'Mask: failed';
    }
  } finally {
    segmentButton.disabled = false;
  }
});
recordButton.addEventListener('click', () => {
  if (front) recordButton.textContent = front.toggleRecord();
  else if (run) recordButton.textContent = run.recorder.toggle() ? '■ Stop' : '● Record';
});
exitButton.addEventListener('click', () => {
  if (front) {
    const summary = front.stop();
    front = null;
    showLanding(summary);
  } else {
    void run?.session.end();
  }
});

void checkSupport();
