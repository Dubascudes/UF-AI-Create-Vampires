import { BackgroundModel, type BackgroundResult } from './background';
import { conservativePersonMask } from './hybrid';
import { bytesToUnit, liveAlphaFromPersonMask, maskCoverage, shouldRequestFill } from './local-compose';
import { YoloStream } from './person-segmentation';
import './style.css';

// Local 2D mirror: webcam → local YOLO person mask → learned room → composite, all on this machine.
// Every displayed frame is the exact frame that was segmented, so the mask never lags the image.
// LaMa fills parts of the room that have never been seen, asynchronously, into the room model.

const PROCESS_WIDTH = 640;

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <main class="shell generated-shell">
    <header class="topbar"><div class="brand"><span class="brand-mark" aria-hidden="true">◌</span><span>ABSENT<span class="brand-separator">/</span> LOCAL 2D MIRROR</span></div><span class="edition">WEBCAM · NO NETWORK</span></header>
    <section class="workspace" aria-label="Mirror workspace"><div class="stage-panel"><div class="stage-topline"><span><span class="live-indicator"></span> <span id="stageLabel">CAMERA OFFLINE</span></span><span id="stageResolution">—</span></div><div class="stage" id="stage"><video id="camera" autoplay muted playsinline aria-hidden="true"></video><canvas id="output" aria-label="Webcam view with people removed"></canvas><div class="stage-empty" id="stageEmpty"><div class="empty-symbol">◌</div><p>Your room appears here</p><span id="emptyHint">Start the camera to begin</span></div></div><div class="stage-footer"><span id="status" role="status" aria-live="polite">Ready.</span><span id="metrics">—</span></div></div>
      <aside class="control-panel" aria-label="Mirror controls"><div class="panel-heading"><span>CONTROL ROOM</span><span class="panel-icon">✳</span></div>
        <div class="object-control"><h2>Start</h2><p>Keep the camera fixed. Move around so the mirror can see the room behind you; parts it has never seen are filled by LaMa.</p><button id="startButton" class="button button-secondary" type="button">Start camera <span aria-hidden="true">↗</span></button></div>
        <div class="object-control"><h2>View</h2><label class="field-label" for="viewMode">Show</label><select class="field-input" id="viewMode"><option value="mirror">Mirror (people removed)</option><option value="debug">Mirror + masks (red person, blue unseen)</option><option value="room">Learned room only</option><option value="raw">Raw camera</option></select></div>
        <label class="toggle-row"><span>Generative fill for unseen areas</span><input id="fillToggle" type="checkbox" role="switch" checked /><span class="switch" aria-hidden="true"></span></label>
        <p class="field-label" id="fillStatus">LaMa fill starts with the camera.</p>
        <button id="resetButton" class="text-button muted" type="button" disabled>Reset learned room</button>
        <button id="stopButton" class="text-button muted" type="button" disabled>Stop camera</button>
      </aside>
    </section>
    <footer class="footer"><span>ABSENT <span class="footer-slash">/</span> MADE TO QUESTION WHAT WE CHOOSE TO SEE</span><span>FRAMES STAY ON THIS COMPUTER</span></footer>
  </main>`;

const select = <T extends Element>(query: string): T => document.querySelector<T>(query)!;
const video = select<HTMLVideoElement>('#camera');
const output = select<HTMLCanvasElement>('#output');
const outputContext = output.getContext('2d')!;
const startButton = select<HTMLButtonElement>('#startButton');
const stopButton = select<HTMLButtonElement>('#stopButton');
const resetButton = select<HTMLButtonElement>('#resetButton');
const viewMode = select<HTMLSelectElement>('#viewMode');
const fillToggle = select<HTMLInputElement>('#fillToggle');
const fillStatus = select<HTMLElement>('#fillStatus');
const status = select<HTMLElement>('#status');
const metrics = select<HTMLElement>('#metrics');
const stageLabel = select<HTMLElement>('#stageLabel');
const stageResolution = select<HTMLElement>('#stageResolution');
const stageEmpty = select<HTMLElement>('#stageEmpty');
const emptyHint = select<HTMLElement>('#emptyHint');

function canvas2d(): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const canvas = document.createElement('canvas');
  return [canvas, canvas.getContext('2d', { willReadFrequently: true })!];
}
const [frameCanvas, frameContext] = canvas2d();
const [roomCanvas, roomContext] = canvas2d();
const [alphaCanvas, alphaContext] = canvas2d();
const [liveCanvas, liveContext] = canvas2d();
const [overlayCanvas, overlayContext] = canvas2d();

let stream: MediaStream | null = null;
let yolo: YoloStream | null = null;
let background: BackgroundModel | null = null;
let previousConfidence: Float32Array | undefined;
let alphaPixels: ImageData | null = null;
let runId = 0;
let sceneGeneration = 0;
let busy = false;
let shown = 0;
let skipped = 0;
let yoloMs: number | null = null;
let stageMs: Record<string, number> = {};
let lastLearned: BackgroundResult | null = null;
let lastPersonCoverage = 0;
let metricsTimer = 0;
const fill = { pending: false, lastRequestAt: -Infinity, requestId: 0 };

function setStatus(message: string, error = false): void {
  status.textContent = message;
  status.classList.toggle('error', error);
}

function resize(width: number, height: number): void {
  for (const canvas of [output, frameCanvas, roomCanvas, alphaCanvas, liveCanvas, overlayCanvas]) {
    canvas.width = width;
    canvas.height = height;
  }
  alphaPixels = alphaContext.createImageData(width, height);
  background = new BackgroundModel(width, height);
  previousConfidence = undefined;
  sceneGeneration++;
  stageResolution.textContent = `${width} × ${height}`;
}

function onVideoFrame(): void {
  if (!stream) return;
  video.requestVideoFrameCallback(onVideoFrame);
  if (busy) { skipped++; return; }
  busy = true;
  void processFrame(runId).finally(() => { busy = false; });
}

async function processFrame(currentRun: number): Promise<void> {
  if (!yolo || !background || !alphaPixels) return;
  const { width, height } = frameCanvas;
  const start = performance.now();
  frameContext.drawImage(video, 0, 0, width, height);
  const frame = frameContext.getImageData(0, 0, width, height);
  const captured = performance.now();
  let detected;
  try {
    detected = await yolo.segment(frame);
  } catch (error) {
    if (currentRun === runId) stopCamera(`Person segmentation stopped: ${String(error)}. Restart npm run segment:serve.`, true);
    return;
  }
  if (currentRun !== runId || !background) return;
  const segmented = performance.now();
  yoloMs = detected.inferenceMs;

  const person = conservativePersonMask(detected.values, width, height, previousConfidence);
  previousConfidence = person.temporal;
  const masked = performance.now();
  const learned = background.observe(frame.data, bytesToUnit(person.mask), width, height);
  const observed = performance.now();
  lastLearned = learned;
  lastPersonCoverage = maskCoverage(person.mask);
  roomContext.putImageData(new ImageData(learned.background, width, height), 0, 0);
  compose(person.mask, learned);
  requestFill(learned, background);
  const done = performance.now();
  stageMs = { capture: captured - start, yolo: segmented - captured, mask: masked - segmented, room: observed - masked, compose: done - observed };
  shown++;
}

function compose(person: Uint8Array, learned: BackgroundResult): void {
  const { width, height } = output;
  const view = viewMode.value;
  if (view === 'raw') { outputContext.drawImage(frameCanvas, 0, 0); return; }
  outputContext.drawImage(roomCanvas, 0, 0);
  if (view === 'room') return;
  liveAlphaFromPersonMask(person, alphaPixels!.data);
  alphaContext.putImageData(alphaPixels!, 0, 0);
  liveContext.globalCompositeOperation = 'source-over';
  liveContext.clearRect(0, 0, width, height);
  liveContext.drawImage(frameCanvas, 0, 0);
  liveContext.globalCompositeOperation = 'destination-in';
  liveContext.filter = 'blur(2px)'; // soften the seam between live camera and learned room
  liveContext.drawImage(alphaCanvas, 0, 0);
  liveContext.filter = 'none';
  outputContext.drawImage(liveCanvas, 0, 0);
  if (view !== 'debug') return;
  const overlay = overlayContext.createImageData(width, height);
  for (let i = 0; i < person.length; i++) {
    const offset = i * 4;
    if (person[i] >= 128) { overlay.data[offset] = 255; overlay.data[offset + 1] = 40; overlay.data[offset + 2] = 80; overlay.data[offset + 3] = 110; }
    else if (learned.missingMask[i]) { overlay.data[offset] = 60; overlay.data[offset + 1] = 120; overlay.data[offset + 2] = 255; overlay.data[offset + 3] = 110; }
  }
  overlayContext.putImageData(overlay, 0, 0);
  outputContext.drawImage(overlayCanvas, 0, 0);
}

function requestFill(learned: BackgroundResult, model: BackgroundModel): void {
  const now = performance.now();
  if (!fillToggle.checked || !shouldRequestFill({ ...fill, ...learned }, now)) return;
  const id = ++fill.requestId;
  const generation = sceneGeneration;
  fill.pending = true;
  fill.lastRequestAt = now;
  fillStatus.textContent = `Filling ${Math.round(learned.missingFraction * 100)}% unseen room with LaMa…`;
  const missing = learned.missingMask;
  const [maskCanvas, maskContext] = canvas2d();
  maskCanvas.width = model.width;
  maskCanvas.height = model.height;
  const maskPixels = maskContext.createImageData(model.width, model.height);
  for (let i = 0; i < missing.length; i++) {
    const offset = i * 4;
    maskPixels.data[offset] = maskPixels.data[offset + 1] = maskPixels.data[offset + 2] = missing[i];
    maskPixels.data[offset + 3] = 255;
  }
  maskContext.putImageData(maskPixels, 0, 0);
  void fetch('/api/inpaint', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: roomCanvas.toDataURL('image/png'), mask: maskCanvas.toDataURL('image/png') }),
  }).then(async response => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const elapsed = response.headers.get('X-Inference-Ms');
    const bitmap = await createImageBitmap(await response.blob());
    try {
      if (id !== fill.requestId || generation !== sceneGeneration || background !== model) return;
      maskContext.drawImage(bitmap, 0, 0, model.width, model.height);
      model.installGenerated(maskContext.getImageData(0, 0, model.width, model.height).data, missing);
      fillStatus.textContent = `LaMa fill installed${elapsed ? ` (${Math.round(Number(elapsed))} ms)` : ''}. Real room pixels replace it when revealed.`;
    } finally {
      bitmap.close();
    }
  }).catch(error => {
    if (id === fill.requestId) fillStatus.textContent = `LaMa unavailable (${String(error)}). Run npm run inpaint:serve. Nearest-color fill remains.`;
  }).finally(() => {
    if (id === fill.requestId) fill.pending = false;
  });
}

function updateMetrics(): void {
  const learned = lastLearned;
  const stages = Object.values(stageMs).map(value => Math.round(value)).join('/');
  metrics.textContent = `SHOWN ${shown} FPS · SKIPPED ${skipped} · YOLO ${yoloMs === null ? '—' : Math.round(yoloMs)} MS · STAGES ${stages || '—'} MS · ROOM ${learned ? Math.round(learned.knownFraction * 100) : '—'}% · PERSON ${Math.round(lastPersonCoverage * 100)}%`;
  metrics.title = 'Stages: capture / YOLO round trip / mask cleanup / room model / compose';
  if (learned && stream) setStatus(learned.knownFraction < 0.95 ? 'Learning the room. Move aside to reveal what is behind you.' : 'Room learned. People are replaced live.');
  shown = skipped = 0;
}

async function startCamera(): Promise<void> {
  if (stream) return;
  const currentRun = ++runId;
  startButton.disabled = true;
  try {
    setStatus('Connecting to the local YOLO service…');
    const health = await fetch('/seg/health');
    if (!health.ok) throw new Error(`YOLO service unavailable (HTTP ${health.status}). Run npm run segment:serve.`);
    const connected = await YoloStream.connect();
    if (currentRun !== runId) { connected.close(); return; }
    yolo = connected;
    setStatus('Requesting camera access…');
    const media = await navigator.mediaDevices.getUserMedia({ audio: false, video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } } });
    if (currentRun !== runId) { media.getTracks().forEach(track => track.stop()); return; }
    stream = media;
    video.srcObject = media;
    await video.play();
    resize(PROCESS_WIDTH, Math.round(PROCESS_WIDTH * video.videoHeight / video.videoWidth));
    stageEmpty.hidden = true;
    stageLabel.textContent = 'VANISHING MIRROR';
    stopButton.disabled = resetButton.disabled = false;
    fillStatus.textContent = 'Waiting for enough of the room to become visible…';
    metricsTimer = window.setInterval(updateMetrics, 1000);
    video.requestVideoFrameCallback(onVideoFrame);
  } catch (error) {
    stopCamera(`Could not start: ${String(error)}`, true);
  }
}

function stopCamera(message = 'Camera stopped.', error = false): void {
  runId++;
  fill.requestId++;
  fill.pending = false;
  window.clearInterval(metricsTimer);
  yolo?.close();
  yolo = null;
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  background = null;
  lastLearned = null;
  outputContext.clearRect(0, 0, output.width, output.height);
  stageEmpty.hidden = false;
  emptyHint.textContent = message;
  stageLabel.textContent = 'CAMERA OFFLINE';
  metrics.textContent = '—';
  startButton.disabled = false;
  stopButton.disabled = resetButton.disabled = true;
  setStatus(message, error);
}

startButton.addEventListener('click', () => { void startCamera(); });
stopButton.addEventListener('click', () => stopCamera());
resetButton.addEventListener('click', () => {
  background?.reset();
  previousConfidence = undefined;
  sceneGeneration++;
  fill.requestId++;
  fill.pending = false;
  fill.lastRequestAt = -Infinity;
  fillStatus.textContent = 'Room reset. Waiting for enough of the room to become visible…';
});
window.addEventListener('beforeunload', () => stream?.getTracks().forEach(track => track.stop()));
