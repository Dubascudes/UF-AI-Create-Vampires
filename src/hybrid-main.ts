import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';
import { BackgroundModel } from './background';
import { composeHybrid, conservativePersonMask, erasePersonForModel } from './hybrid';
import { ObjectTracker } from './object-tracker';
import { OutputQueue } from './output-queue';
import { YoloStream, type PersonMask } from './person-segmentation';
import { createStartMessage, shouldDisplayOutput, shouldSendInputFrame, type OutputMetadata } from './joyai-protocol';
import './style.css';

const MODES = {
  b200: { width: 1248, height: 720, fps: 30, localPort: 8083 },
  rtx6000: { width: 840, height: 480, fps: 24, localPort: 8084 },
} as const;
let mode: { width: number; height: number; fps: number; localPort: number } = MODES.b200;
const PROMPT = 'Remove every person from the video. Keep the room, lighting, furniture, and objects in place. Photorealistic stable camera.';

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <main class="shell generated-shell">
    <header class="topbar"><div class="brand"><span class="brand-mark" aria-hidden="true">◌</span><span>ABSENT<span class="brand-separator">/</span> A LIVE VANISHING MIRROR</span></div><span class="edition">GATOR CREATE · 2026</span></header>
    <section class="hero" aria-labelledby="title"><div class="hero-copy"><p class="eyebrow">HYBRID LIVE MIRROR</p><h1 id="title">What does a room<br />look like <em>without us?</em></h1><p class="intro">The mirror learns the room as it becomes visible, then uses a video model to fill places the camera has never seen.</p><div class="hero-actions"><button id="startButton" class="button button-primary" type="button">Start mirror <span aria-hidden="true">↗</span></button><span class="privacy-note"><span class="privacy-dot"></span> Camera frames go to the configured model endpoint</span></div></div><div class="hero-index" aria-hidden="true"><span>01 / 01</span><span>THE OBSERVATION</span></div></section>
    <section class="workspace" aria-label="Mirror workspace"><div class="stage-panel"><div class="stage-topline"><span><span class="live-indicator"></span> <span id="stageLabel">MODEL DISCONNECTED</span></span><span id="stageResolution">1248 × 720</span></div><div class="stage" id="stage"><video id="camera" autoplay muted playsinline aria-hidden="true"></video><canvas id="output" width="1248" height="720" aria-label="Person-removed webcam view"></canvas><div class="stage-empty" id="stageEmpty"><div class="empty-symbol">◌</div><p>Your room appears here</p><span id="emptyHint">Start the camera and connect to the model</span></div></div><div class="stage-footer"><span id="status" role="status" aria-live="polite">Ready to connect.</span><span id="metrics">OUTPUT — · DELAY — · ROOM —</span></div></div>
      <aside class="control-panel" aria-label="Mirror controls"><div class="panel-heading"><span>CONTROL ROOM</span><span class="panel-icon">✳</span></div><div class="object-control"><h2>Model endpoint</h2><p>Forward the HiPerGator service to a local port, then connect here.</p><label class="field-label" for="mode">Serving profile</label><select class="field-input" id="mode"><option value="b200">B200 · 720p / 30 fps</option><option value="rtx6000">RTX Pro 6000 · 480p / 24 fps</option></select><label class="field-label" for="endpoint">WebSocket URL</label><input class="field-input" id="endpoint" type="url" value="ws://127.0.0.1:8083/ws" spellcheck="false" /></div><div class="object-control"><h2>Person segmentation</h2><p>Compare the local GPU model with the original browser segmenter.</p><label class="field-label" for="segmenterMode">Mask model</label><select class="field-input" id="segmenterMode"><option value="yolo">YOLO26n-seg · local GPU</option><option value="mediapipe">MediaPipe Selfie Segmenter · CPU</option></select><a class="preview-link" href="/mask-preview.html">Compare live masks ↗</a></div><div class="object-control"><h2>Edit instruction</h2><p>The model uses this instruction across the live stream.</p><label class="field-label" for="prompt">Prompt</label><textarea class="field-input field-textarea" id="prompt" rows="4"></textarea></div><div class="object-control"><h2>Learned room</h2><p>Keep the camera fixed. Move aside to reveal hidden areas; the mirror remembers them. Generated pixels are used only in masked areas that remain unseen.</p><button id="resetButton" class="text-button muted" type="button" disabled>Reset learned room</button></div><div class="object-control"><h2>Make an object float</h2><p>Select an object, then click its visible surface in the mirror.</p><button id="objectButton" class="button button-secondary" type="button" disabled>Choose object <span aria-hidden="true">↗</span></button><button id="clearObjectButton" class="text-button muted" type="button" hidden>Clear object</button><p id="objectStatus" role="status">Loads with the camera.</p></div><button id="stopButton" class="text-button muted" type="button" disabled>Stop mirror</button></aside>
    </section><footer class="footer"><span>ABSENT <span class="footer-slash">/</span> MADE TO QUESTION WHAT WE CHOOSE TO SEE</span><span>LIVE CAMERA FRAMES SENT TO YOUR MODEL ENDPOINT</span></footer>
  </main>`;

const select = <T extends Element>(query: string): T => document.querySelector<T>(query)!;
const video = select<HTMLVideoElement>('#camera');
const output = select<HTMLCanvasElement>('#output');
const stage = select<HTMLDivElement>('#stage');
const outputContext = output.getContext('2d')!;
const capture = document.createElement('canvas');
const captureContext = capture.getContext('2d', { willReadFrequently: true })!;
const small = document.createElement('canvas');
const smallContext = small.getContext('2d', { willReadFrequently: true })!;
const outputSmall = document.createElement('canvas');
const outputSmallContext = outputSmall.getContext('2d', { willReadFrequently: true })!;
const roomCanvas = document.createElement('canvas');
const roomContext = roomCanvas.getContext('2d')!;
const roomUpscale = document.createElement('canvas');
const roomUpscaleContext = roomUpscale.getContext('2d', { willReadFrequently: true })!;
const generatedCanvas = document.createElement('canvas');
const generatedContext = generatedCanvas.getContext('2d', { willReadFrequently: true })!;
const inputCanvas = document.createElement('canvas');
const inputContext = inputCanvas.getContext('2d')!;
const startButton = select<HTMLButtonElement>('#startButton');
const stopButton = select<HTMLButtonElement>('#stopButton');
const resetButton = select<HTMLButtonElement>('#resetButton');
const objectButton = select<HTMLButtonElement>('#objectButton');
const clearObjectButton = select<HTMLButtonElement>('#clearObjectButton');
const objectStatus = select<HTMLElement>('#objectStatus');
const endpoint = select<HTMLInputElement>('#endpoint');
const modeSelect = select<HTMLSelectElement>('#mode');
const segmenterSelect = select<HTMLSelectElement>('#segmenterMode');
const prompt = select<HTMLTextAreaElement>('#prompt');
const status = select<HTMLElement>('#status');
const metrics = select<HTMLElement>('#metrics');
const stageLabel = select<HTMLElement>('#stageLabel');
const stageResolution = select<HTMLElement>('#stageResolution');
const stageEmpty = select<HTMLElement>('#stageEmpty');
const emptyHint = select<HTMLElement>('#emptyHint');
prompt.value = PROMPT;

interface SentFrame { raw: ImageData; smallPerson: Uint8Array; roomSmall: Uint8ClampedArray; knownSmall: Uint8Array; missing: Uint8Array; objectSmall?: Uint8Array; capturedAt: number; scene: number }
const sent = new Map<number, SentFrame>();
let stream: MediaStream | null = null;
let socket: WebSocket | null = null;
let yoloStream: YoloStream | null = null;
let inputSegmenter: ImageSegmenter | null = null;
let outputSegmenter: ImageSegmenter | null = null;
let background: BackgroundModel | null = null;
let previousConfidence: Float32Array | undefined;
let segmentationMode: 'yolo' | 'mediapipe' = 'yolo';
let maskInferenceMs: number | null = null;
let inputProcessingMs: number | null = null;
let outputProcessingMs: number | null = null;
let sendTimer = 0;
let metricsTimer = 0;
let busy = false;
let sentFrames = 0;
let receivedFrames = 0;
let receivedPackets = 0;
let lastCaptureMs = 0;
let latencyMs: number | null = null;
let lastSentAt = 0;
let pendingMeta: OutputMetadata | null = null;
let runId = 0;
let objectWorker: Worker | null = null;
let objectReady = false;
let selectingObject = false;
let selectionId = 0;
let selectionFrame: Uint8ClampedArray | null = null;
let objectTracker: ObjectTracker | null = null;
let lostObjectFrames = 0;
let sceneGeneration = 0;
const outputQueue = new OutputQueue<{ data: ArrayBuffer; meta: OutputMetadata | null; run: number }>(
  async item => displayFrame(item.data, item.meta, item.run), 8,
  error => setStatus(`Generated frame failed: ${String(error)}`, true),
);

function setStatus(message: string, error = false): void {
  status.textContent = message;
  status.classList.toggle('error', error);
}

async function segmentPersonFrame(canvas: HTMLCanvasElement, role: 'input' | 'output', pixels?: ImageData): Promise<PersonMask> {
  if (segmentationMode === 'yolo') {
    if (!yoloStream || !pixels) throw new Error('Local YOLO stream is unavailable');
    return yoloStream.segment(pixels);
  }
  const segmenter = role === 'input' ? inputSegmenter : outputSegmenter;
  if (!segmenter) throw new Error('MediaPipe person segmenter is unavailable');
  const start = performance.now();
  const result = role === 'input' ? segmenter.segmentForVideo(canvas, start) : segmenter.segment(canvas);
  const mask = result.confidenceMasks?.[1] ?? result.confidenceMasks?.[0];
  if (!mask) throw new Error('Person segmenter returned no confidence mask');
  return { values: mask.getAsFloat32Array().slice(), width: mask.width, height: mask.height,
    inferenceMs: performance.now() - start };
}

function resizeBytes(source: Uint8Array, width: number, height: number, targetWidth: number, targetHeight: number): Uint8Array {
  const result = new Uint8Array(targetWidth * targetHeight);
  for (let y = 0; y < targetHeight; y++) {
    const sy = Math.min(height - 1, Math.floor(y * height / targetHeight));
    for (let x = 0; x < targetWidth; x++) result[y * targetWidth + x] = source[sy * width + Math.min(width - 1, Math.floor(x * width / targetWidth))];
  }
  return result;
}

function resizeObjectMask(source: Float32Array, width: number, height: number, targetWidth: number, targetHeight: number): Float32Array {
  const result = new Float32Array(targetWidth * targetHeight);
  for (let y = 0; y < targetHeight; y++) {
    const sy = Math.min(height - 1, Math.floor(y * height / targetHeight));
    for (let x = 0; x < targetWidth; x++) result[y * targetWidth + x] = source[sy * width + Math.min(width - 1, Math.floor(x * width / targetWidth))];
  }
  return result;
}

function clearObject(): void {
  objectTracker = null;
  lostObjectFrames = 0;
  selectionFrame = null;
  clearObjectButton.hidden = true;
  objectButton.textContent = 'Choose object ↗';
}

function startObjectWorker(): void {
  const worker = new Worker('/object-worker.js');
  objectWorker = worker;
  worker.onmessage = (event: MessageEvent) => {
    if (worker !== objectWorker) return;
    const message = event.data as { type: string; id?: number; width?: number; height?: number; values?: Float32Array; message?: string };
    if (message.type === 'ready') {
      objectReady = true;
      objectButton.disabled = false;
      objectStatus.textContent = 'Choose an object, then click it in the mirror.';
    } else if (message.type === 'selected' && message.id === selectionId && message.values && message.width && message.height && selectionFrame) {
      const mask = resizeObjectMask(message.values, message.width, message.height, small.width, small.height);
      const tracker = ObjectTracker.create(selectionFrame, mask, small.width, small.height);
      if (!tracker) {
        objectStatus.textContent = 'Could not isolate that object. Try another point.';
      } else {
        objectTracker = tracker;
        background?.forget(mask, small.width, small.height);
        clearObjectButton.hidden = false;
        objectButton.textContent = 'Choose another object ↗';
        objectStatus.textContent = 'Object kept visible. Keep it facing the camera.';
      }
      selectionFrame = null;
      objectButton.disabled = false;
    } else if (message.type === 'error') {
      if (message.id !== undefined && message.id !== selectionId) return;
      selectionFrame = null;
      objectButton.disabled = !objectReady;
      objectStatus.textContent = `Object selection failed: ${message.message ?? 'Unknown error'}`;
    }
  };
  worker.onerror = () => { objectReady = false; objectButton.disabled = true; objectStatus.textContent = 'Object selection could not start.'; };
  worker.postMessage({ type: 'init' });
}

function drawCapture(): void {
  const sw = video.videoWidth;
  const sh = video.videoHeight;
  const aspect = mode.width / mode.height;
  if (sw / sh > aspect) {
    const width = sh * aspect;
    captureContext.drawImage(video, (sw - width) / 2, 0, width, sh, 0, 0, mode.width, mode.height);
  } else {
    const height = sw / aspect;
    captureContext.drawImage(video, 0, (sh - height) / 2, sw, height, 0, 0, mode.width, mode.height);
  }
}

async function sendFrame(currentRun: number): Promise<void> {
  const activeSocket = socket;
  if (busy || !activeSocket || activeSocket.readyState !== WebSocket.OPEN || !stream || !background || video.readyState < 2) return;
  if (!shouldSendInputFrame(sentFrames, receivedPackets, lastSentAt, performance.now()) ||
      sentFrames - receivedPackets >= 24 || activeSocket.bufferedAmount > 1_000_000) return;
  busy = true;
  const processingStart = performance.now();
  try {
    drawCapture();
    const capturedAt = Date.now();
    const raw = captureContext.getImageData(0, 0, mode.width, mode.height);
    smallContext.drawImage(capture, 0, 0, small.width, small.height);
    const smallFrame = smallContext.getImageData(0, 0, small.width, small.height);
    if (objectTracker && !objectTracker.update(smallFrame.data)) {
      if (++lostObjectFrames >= 5) { clearObject(); objectStatus.textContent = 'Object tracking was lost. Choose it again.'; }
    } else lostObjectFrames = 0;
    const detected = await segmentPersonFrame(small, 'input', smallFrame);
    if (currentRun !== runId) return;
    maskInferenceMs = detected.inferenceMs;
    const conservative = conservativePersonMask(detected.values, detected.width, detected.height, previousConfidence);
    previousConfidence = conservative.temporal;
    const smallPerson = resizeBytes(conservative.mask, detected.width, detected.height, small.width, small.height);
    const tracked = objectTracker?.projectMask(small.width, small.height);
    const learned = background.observe(smallFrame.data, Float32Array.from(smallPerson, value => value / 255), small.width, small.height, tracked);
    roomContext.putImageData(new ImageData(learned.background, small.width, small.height), 0, 0);
    inputContext.imageSmoothingEnabled = true;
    inputContext.drawImage(roomCanvas, 0, 0, mode.width, mode.height);
    const room = inputContext.getImageData(0, 0, mode.width, mode.height).data;
    const person = resizeBytes(smallPerson, small.width, small.height, mode.width, mode.height);
    const objectSmall = tracked ? Uint8Array.from(tracked, value => value >= 0.5 ? 255 : 0) : undefined;
    const object = objectSmall ? resizeBytes(objectSmall, small.width, small.height, mode.width, mode.height) : undefined;
    inputContext.putImageData(new ImageData(erasePersonForModel(raw.data, room, person, object), mode.width, mode.height), 0, 0);
    const blob = await new Promise<Blob | null>(resolve => inputCanvas.toBlob(resolve, 'image/jpeg', 0.78));
    if (!blob || currentRun !== runId || socket !== activeSocket || activeSocket.readyState !== WebSocket.OPEN) return;
    const seq = ++sentFrames;
    sent.set(seq, { raw, smallPerson, roomSmall: learned.background, knownSmall: learned.knownMask,
      missing: learned.missingMask, objectSmall, capturedAt, scene: sceneGeneration });
    while (sent.size > 32) sent.delete(sent.keys().next().value!);
    activeSocket.send(JSON.stringify({ type: 'frame_meta', seq, t_capture_ms: capturedAt }));
    activeSocket.send(blob);
    lastSentAt = performance.now();
    inputProcessingMs = lastSentAt - processingStart;
  } catch (error) {
    if (currentRun === runId) setStatus(`Frame processing failed: ${String(error)}`, true);
  } finally {
    busy = false;
  }
}

async function displayFrame(data: ArrayBuffer, meta: OutputMetadata | null, currentRun: number): Promise<void> {
  if (!shouldDisplayOutput(meta, lastCaptureMs)) return;
  const frame = sent.get(Number(meta!.source_seq));
  if (!frame) return;
  const processingStart = performance.now();
  const bitmap = await createImageBitmap(new Blob([data], { type: 'image/jpeg' }));
    try {
      if (currentRun !== runId || !shouldDisplayOutput(meta, lastCaptureMs)) return;
      generatedContext.drawImage(bitmap, 0, 0, mode.width, mode.height);
      const generated = generatedContext.getImageData(0, 0, mode.width, mode.height);
      const raw = frame.raw.data;
      outputSmallContext.drawImage(bitmap, 0, 0, outputSmall.width, outputSmall.height);
      const smallGenerated = outputSmallContext.getImageData(0, 0, outputSmall.width, outputSmall.height);
      const detected = await segmentPersonFrame(outputSmall, 'output', smallGenerated);
      if (currentRun !== runId || !shouldDisplayOutput(meta, lastCaptureMs)) return;
      const smallGeneratedPerson = resizeBytes(
        conservativePersonMask(detected.values, detected.width, detected.height).mask,
        detected.width, detected.height, outputSmall.width, outputSmall.height,
      );
      const generatedPerson = resizeBytes(smallGeneratedPerson, outputSmall.width, outputSmall.height, mode.width, mode.height);
      roomContext.putImageData(new ImageData(frame.roomSmall, small.width, small.height), 0, 0);
      roomUpscaleContext.drawImage(roomCanvas, 0, 0, mode.width, mode.height);
      const room = roomUpscaleContext.getImageData(0, 0, mode.width, mode.height).data;
      const person = resizeBytes(frame.smallPerson, small.width, small.height, mode.width, mode.height);
      const known = resizeBytes(frame.knownSmall, small.width, small.height, mode.width, mode.height);
      const object = frame.objectSmall
        ? resizeBytes(frame.objectSmall, small.width, small.height, mode.width, mode.height) : undefined;
      if (generatedPerson) for (let i = 0; i < person.length; i++) person[i] = Math.max(person[i], generatedPerson[i]);
      const composite = composeHybrid(raw, room, generated.data, {
        person, known, generatedPerson, object,
      });
      outputContext.putImageData(new ImageData(composite, mode.width, mode.height), 0, 0);
      if (background && frame.scene === sceneGeneration) {
        const safeMissing = frame.missing.slice();
        for (let i = 0; i < safeMissing.length; i++) {
          if (frame.smallPerson[i] < 128 || smallGeneratedPerson[i] >= 128) safeMissing[i] = 0;
        }
        background.installGenerated(smallGenerated.data, safeMissing);
      }
      const displayedSeq = Number(meta!.source_seq);
      for (const seq of sent.keys()) if (seq <= displayedSeq) sent.delete(seq);
      lastCaptureMs = Number(meta!.t_capture_ms);
      latencyMs = Date.now() - frame.capturedAt;
      receivedFrames++;
      outputProcessingMs = performance.now() - processingStart;
      stageEmpty.hidden = true;
      stageLabel.textContent = 'VANISHING MIRROR';
      setStatus(background && background.knownFraction < 0.95 ? 'Learning the room. Move aside to reveal hidden areas.' : 'Learned room active.');
    } finally { bitmap.close(); }
}

function stopMirror(message = 'Mirror stopped.'): void {
  runId++;
  window.clearInterval(sendTimer);
  window.clearInterval(metricsTimer);
  const oldSocket = socket;
  socket = null;
  if (oldSocket?.readyState === WebSocket.OPEN) oldSocket.send(JSON.stringify({ type: 'stop' }));
  oldSocket?.close();
  yoloStream?.close();
  yoloStream = null;
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  inputSegmenter?.close();
  outputSegmenter?.close();
  inputSegmenter = outputSegmenter = null;
  background = null;
  sceneGeneration++;
  previousConfidence = undefined;
  sent.clear();
  busy = false;
  outputQueue.clear();
  sentFrames = receivedFrames = receivedPackets = lastCaptureMs = lastSentAt = 0;
  latencyMs = null;
  maskInferenceMs = null;
  inputProcessingMs = outputProcessingMs = null;
  pendingMeta = null;
  objectWorker?.terminate();
  objectWorker = null;
  objectReady = false;
  selectingObject = false;
  selectionId++;
  clearObject();
  stage.classList.remove('is-selecting');
  objectButton.disabled = true;
  objectStatus.textContent = 'Loads with the camera.';
  outputContext.clearRect(0, 0, mode.width, mode.height);
  stageEmpty.hidden = false;
  emptyHint.textContent = message;
  stageLabel.textContent = 'MODEL DISCONNECTED';
  metrics.textContent = 'OUTPUT — · DELAY — · ROOM —';
  startButton.disabled = false;
  stopButton.disabled = resetButton.disabled = true;
  endpoint.disabled = modeSelect.disabled = segmenterSelect.disabled = prompt.disabled = false;
  setStatus(message);
}

async function startMirror(): Promise<void> {
  if (stream || socket) return;
  const url = endpoint.value.trim();
  if (!/^wss?:\/\//i.test(url)) { setStatus('Enter a ws:// or wss:// model endpoint.', true); return; }
  const currentRun = ++runId;
  mode = MODES[modeSelect.value as keyof typeof MODES] ?? MODES.b200;
  segmentationMode = segmenterSelect.value === 'mediapipe' ? 'mediapipe' : 'yolo';
  for (const canvas of [capture, generatedCanvas, inputCanvas, output, roomUpscale]) { canvas.width = mode.width; canvas.height = mode.height; }
  small.width = roomCanvas.width = Math.round(mode.width / 3);
  small.height = roomCanvas.height = Math.round(mode.height / 3);
  outputSmall.width = small.width;
  outputSmall.height = small.height;
  background = new BackgroundModel(small.width, small.height);
  sceneGeneration++;
  stageResolution.textContent = `${mode.width} × ${mode.height}`;
  startButton.disabled = true;
  endpoint.disabled = modeSelect.disabled = segmenterSelect.disabled = prompt.disabled = true;
  setStatus('Requesting camera access…');
  try {
    if (segmentationMode === 'yolo') {
      setStatus('Checking the local YOLO service…');
      const response = await fetch('/seg/health');
      if (!response.ok) throw new Error(`Local YOLO service is unavailable (HTTP ${response.status}). Run npm run segment:serve.`);
      const connected = await YoloStream.connect();
      if (currentRun !== runId) { connected.close(); return; }
      yoloStream = connected;
    }
    stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' } });
    if (currentRun !== runId) return;
    video.srcObject = stream;
    await video.play();
    if (currentRun !== runId) return;
    if (segmentationMode === 'mediapipe') {
      setStatus('Loading MediaPipe person segmentation…');
      const vision = await FilesetResolver.forVisionTasks('/wasm');
      inputSegmenter = await ImageSegmenter.createFromOptions(vision, {
        baseOptions: { modelAssetPath: '/models/selfie_segmenter.tflite', delegate: 'CPU' },
        runningMode: 'VIDEO', outputCategoryMask: false, outputConfidenceMasks: true,
      });
      outputSegmenter = await ImageSegmenter.createFromOptions(vision, {
        baseOptions: { modelAssetPath: '/models/selfie_segmenter.tflite', delegate: 'CPU' },
        runningMode: 'IMAGE', outputCategoryMask: false, outputConfidenceMasks: true,
      });
    }
    if (currentRun !== runId) return;
    startObjectWorker();
    const currentSocket = new WebSocket(url);
    currentSocket.binaryType = 'arraybuffer';
    socket = currentSocket;
    stopButton.disabled = resetButton.disabled = false;
    setStatus('Connecting to the generative model…');
    currentSocket.onopen = () => { if (currentRun === runId) setStatus('Waiting for a model session…'); };
    currentSocket.onmessage = event => {
      if (currentRun !== runId || currentSocket !== socket) return;
      if (typeof event.data === 'string') {
        let message: Record<string, unknown>;
        try { message = JSON.parse(event.data) as Record<string, unknown>; }
        catch { setStatus('The model sent an invalid response.', true); return; }
        if (message.type === 'session_granted') currentSocket.send(JSON.stringify(createStartMessage(prompt.value.trim(), mode.width, mode.height, mode.fps)));
        else if (message.type === 'queue_position') setStatus(`Waiting for model session · ${Number(message.ahead ?? message.position ?? 0)} ahead`);
        else if (message.type === 'started') {
          setStatus('Model started. Learning the room…');
          let previousCount = 0;
          let previousSent = 0;
          let previousPackets = 0;
          sendTimer = window.setInterval(() => { void sendFrame(currentRun); }, 1000 / mode.fps);
          metricsTimer = window.setInterval(() => {
            const fps = receivedFrames - previousCount;
            const inputFps = sentFrames - previousSent;
            const packetFps = receivedPackets - previousPackets;
            previousCount = receivedFrames;
            previousSent = sentFrames;
            previousPackets = receivedPackets;
            metrics.textContent = `IN ${inputFps} · MODEL ${packetFps} · SHOWN ${fps} FPS · DELAY ${latencyMs === null ? '—' : `${latencyMs} MS`} · WORK ${inputProcessingMs === null ? '—' : `${Math.round(inputProcessingMs)}/${Math.round(outputProcessingMs ?? 0)} MS`}`;
          }, 1000);
        } else if (message.type === 'output_frame') pendingMeta = message as OutputMetadata;
        else if (message.type === 'chunk_done') currentSocket.send(JSON.stringify({ type: 'ack', recv: receivedPackets }));
        else if (message.type === 'error') { setStatus(`Model error: ${String(message.message ?? 'unknown error')}`, true); window.clearInterval(sendTimer); }
      } else if (event.data instanceof ArrayBuffer) {
        receivedPackets++;
        const meta = pendingMeta;
        pendingMeta = null;
        outputQueue.enqueue({ data: event.data, meta, run: currentRun });
      }
    };
    currentSocket.onerror = () => { if (currentRun === runId) setStatus('Could not connect to the model endpoint.', true); };
    currentSocket.onclose = () => { if (currentRun === runId) stopMirror('Model connection closed.'); };
  } catch (error) {
    stopMirror('Camera or model connection failed.');
    setStatus(`Could not start: ${String(error)}`, true);
  }
}

modeSelect.addEventListener('change', () => {
  const selected = MODES[modeSelect.value as keyof typeof MODES];
  if (selected) endpoint.value = `ws://127.0.0.1:${selected.localPort}/ws`;
});
startButton.addEventListener('click', () => { void startMirror(); });
stopButton.addEventListener('click', () => stopMirror());
resetButton.addEventListener('click', () => { background?.reset(); sceneGeneration++; sent.clear(); setStatus('Room reset. Learning starts with the next frame.'); });
objectButton.addEventListener('click', () => {
  if (!objectReady) return;
  selectingObject = !selectingObject;
  stage.classList.toggle('is-selecting', selectingObject);
  objectStatus.textContent = selectingObject ? 'Click an uncovered part of the object.' : 'Object selection cancelled.';
});
output.addEventListener('click', async event => {
  if (!selectingObject || !objectWorker || !objectReady || !stream) return;
  selectingObject = false;
  stage.classList.remove('is-selecting');
  objectButton.disabled = true;
  objectStatus.textContent = 'Isolating the selected object…';
  const rect = output.getBoundingClientRect();
  const x = Math.max(0, Math.min(1, 1 - (event.clientX - rect.left) / rect.width));
  const y = Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height));
  const id = ++selectionId;
  try {
    drawCapture();
    smallContext.drawImage(capture, 0, 0, small.width, small.height);
    selectionFrame = smallContext.getImageData(0, 0, small.width, small.height).data;
    const bitmap = await createImageBitmap(small);
    if (!objectWorker || id !== selectionId) { bitmap.close(); return; }
    objectWorker.postMessage({ type: 'select', id, bitmap, x, y }, [bitmap]);
  } catch (error) {
    objectButton.disabled = false;
    objectStatus.textContent = `Could not capture the object: ${String(error)}`;
  }
});
clearObjectButton.addEventListener('click', () => {
  selectionId++;
  clearObject();
  objectStatus.textContent = 'Object cleared.';
});
window.addEventListener('beforeunload', () => stream?.getTracks().forEach(track => track.stop()));
