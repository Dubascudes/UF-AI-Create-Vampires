import { createStartMessage, shouldDisplayOutput, shouldSendInputFrame, type OutputMetadata } from './joyai-protocol';
import './style.css';

const MODES = {
  b200: { width: 1248, height: 720, fps: 30 },
  rtx6000: { width: 840, height: 480, fps: 24 },
} as const;
let mode: { width: number; height: number; fps: number } = MODES.b200;
const DEFAULT_PROMPT = 'Remove every person from the video. Keep the room, lighting, furniture, and objects in place. Photorealistic stable camera.';

const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `
  <main class="shell generated-shell">
    <header class="topbar">
      <div class="brand"><span class="brand-mark" aria-hidden="true">◌</span><span>ABSENT<span class="brand-separator">/</span> A LIVE VANISHING MIRROR</span></div>
      <span class="edition">GATOR CREATE · 2026</span>
    </header>
    <section class="hero" aria-labelledby="title">
      <div class="hero-copy">
        <p class="eyebrow">GENERATED LIVE</p>
        <h1 id="title">What does a room<br />look like <em>without us?</em></h1>
        <p class="intro">Each displayed frame comes from a streaming video model editing the live camera view.</p>
        <div class="hero-actions">
          <button id="startButton" class="button button-primary" type="button">Start generated mirror <span aria-hidden="true">↗</span></button>
          <span class="privacy-note"><span class="privacy-dot"></span> Camera frames go to the configured model endpoint</span>
        </div>
      </div>
      <div class="hero-index" aria-hidden="true"><span>01 / 01</span><span>THE OBSERVATION</span></div>
    </section>
    <section class="workspace" aria-label="Generated mirror workspace">
      <div class="stage-panel">
        <div class="stage-topline"><span><span class="live-indicator"></span> <span id="stageLabel">MODEL DISCONNECTED</span></span><span id="stageResolution">1248 × 720</span></div>
        <div class="stage">
          <video id="camera" autoplay muted playsinline aria-hidden="true"></video>
          <canvas id="output" width="1248" height="720" aria-label="Generated webcam view"></canvas>
          <div class="stage-empty" id="stageEmpty"><div class="empty-symbol">◌</div><p>Generated frames appear here</p><span id="emptyHint">Start the camera and connect to the model</span></div>
        </div>
        <div class="stage-footer"><span id="status" role="status" aria-live="polite">Ready to connect.</span><span id="metrics">OUTPUT — · DELAY —</span></div>
      </div>
      <aside class="control-panel" aria-label="Mirror controls">
        <div class="panel-heading"><span>CONTROL ROOM</span><span class="panel-icon">✳</span></div>
        <div class="object-control"><h2>Model endpoint</h2><p>Forward the HiPerGator service to local port 8080, then connect here.</p><label class="field-label" for="mode">Serving profile</label><select class="field-input" id="mode"><option value="b200">B200 · 720p / 30 fps</option><option value="rtx6000">RTX Pro 6000 · 480p / 24 fps</option></select><label class="field-label" for="endpoint">WebSocket URL</label><input class="field-input" id="endpoint" type="url" value="ws://127.0.0.1:8080/ws" spellcheck="false" /></div>
        <div class="object-control"><h2>Edit instruction</h2><p>The model uses this instruction across the live stream.</p><label class="field-label" for="prompt">Prompt</label><textarea class="field-input field-textarea" id="prompt" rows="5"></textarea></div>
        <div class="object-control"><h2>Generated output only</h2><p>The raw camera image is used as model input and is never shown in the mirror. If the model pauses or disconnects, output pauses.</p></div>
        <button id="stopButton" class="text-button muted" type="button" disabled>Stop mirror</button>
      </aside>
    </section>
    <footer class="footer"><span>ABSENT <span class="footer-slash">/</span> MADE TO QUESTION WHAT WE CHOOSE TO SEE</span><span>LIVE CAMERA FRAMES SENT TO YOUR MODEL ENDPOINT</span></footer>
  </main>`;

const video = document.querySelector<HTMLVideoElement>('#camera')!;
const output = document.querySelector<HTMLCanvasElement>('#output')!;
const outputContext = output.getContext('2d')!;
const capture = document.createElement('canvas');
capture.width = mode.width;
capture.height = mode.height;
const captureContext = capture.getContext('2d')!;
const startButton = document.querySelector<HTMLButtonElement>('#startButton')!;
const stopButton = document.querySelector<HTMLButtonElement>('#stopButton')!;
const endpoint = document.querySelector<HTMLInputElement>('#endpoint')!;
const modeSelect = document.querySelector<HTMLSelectElement>('#mode')!;
const prompt = document.querySelector<HTMLTextAreaElement>('#prompt')!;
const status = document.querySelector<HTMLElement>('#status')!;
const metrics = document.querySelector<HTMLElement>('#metrics')!;
const stageLabel = document.querySelector<HTMLElement>('#stageLabel')!;
const stageResolution = document.querySelector<HTMLElement>('#stageResolution')!;
const stageEmpty = document.querySelector<HTMLElement>('#stageEmpty')!;
const emptyHint = document.querySelector<HTMLElement>('#emptyHint')!;
prompt.value = DEFAULT_PROMPT;

let stream: MediaStream | null = null;
let socket: WebSocket | null = null;
let sendTimer = 0;
let metricsTimer = 0;
let captureBusy = false;
let sentFrames = 0;
let receivedFrames = 0;
let receivedPackets = 0;
let lastCaptureMs = 0;
let lastLatencyMs: number | null = null;
let lastSentAt = 0;
let pendingOutputMeta: OutputMetadata | null = null;
let runId = 0;

function setStatus(message: string, error = false): void {
  status.textContent = message;
  status.classList.toggle('error', error);
}

function clearTimers(): void {
  window.clearInterval(sendTimer);
  window.clearInterval(metricsTimer);
  sendTimer = 0;
  metricsTimer = 0;
}

function stopMirror(message = 'Mirror stopped.'): void {
  runId++;
  clearTimers();
  const oldSocket = socket;
  socket = null;
  if (oldSocket && oldSocket.readyState === WebSocket.OPEN) oldSocket.send(JSON.stringify({ type: 'stop' }));
  oldSocket?.close();
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  captureBusy = false;
  pendingOutputMeta = null;
  sentFrames = 0;
  receivedFrames = 0;
  receivedPackets = 0;
  lastCaptureMs = 0;
  lastLatencyMs = null;
  lastSentAt = 0;
  outputContext.clearRect(0, 0, mode.width, mode.height);
  stageEmpty.hidden = false;
  emptyHint.textContent = message;
  stageLabel.textContent = 'MODEL DISCONNECTED';
  metrics.textContent = 'OUTPUT — · DELAY —';
  startButton.disabled = false;
  stopButton.disabled = true;
  endpoint.disabled = false;
  modeSelect.disabled = false;
  prompt.disabled = false;
  setStatus(message);
}

function drawCaptureFrame(): void {
  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;
  const sourceAspect = sourceWidth / sourceHeight;
  const targetAspect = mode.width / mode.height;
  if (sourceAspect > targetAspect) {
    const cropWidth = sourceHeight * targetAspect;
    captureContext.drawImage(video, (sourceWidth - cropWidth) / 2, 0, cropWidth, sourceHeight, 0, 0, mode.width, mode.height);
  } else {
    const cropHeight = sourceWidth / targetAspect;
    captureContext.drawImage(video, 0, (sourceHeight - cropHeight) / 2, sourceWidth, cropHeight, 0, 0, mode.width, mode.height);
  }
}

async function sendFrame(currentRun: number): Promise<void> {
  const currentSocket = socket;
  if (captureBusy || !currentSocket || currentSocket.readyState !== WebSocket.OPEN || !stream || video.readyState < 2) return;
  if (!shouldSendInputFrame(sentFrames, receivedPackets, lastSentAt, performance.now())) return;
  if (currentSocket.bufferedAmount > 1_000_000) return;
  captureBusy = true;
  try {
    drawCaptureFrame();
    const capturedAt = Date.now();
    const blob = await new Promise<Blob | null>(resolve => capture.toBlob(resolve, 'image/jpeg', 0.78));
    if (!blob || currentRun !== runId || currentSocket !== socket || currentSocket.readyState !== WebSocket.OPEN) return;
    const seq = ++sentFrames;
    currentSocket.send(JSON.stringify({ type: 'frame_meta', seq, t_capture_ms: capturedAt }));
    currentSocket.send(blob);
    lastSentAt = performance.now();
  } catch (error) {
    setStatus(`Could not send a camera frame: ${String(error)}`, true);
  } finally {
    captureBusy = false;
  }
}

async function displayGeneratedFrame(data: ArrayBuffer, meta: OutputMetadata | null, currentRun: number): Promise<void> {
  if (!shouldDisplayOutput(meta, lastCaptureMs)) return;
  const bitmap = await createImageBitmap(new Blob([data], { type: 'image/jpeg' }));
  try {
    if (currentRun !== runId || !shouldDisplayOutput(meta, lastCaptureMs)) return;
    outputContext.drawImage(bitmap, 0, 0, mode.width, mode.height);
    lastCaptureMs = Number(meta!.t_capture_ms);
    lastLatencyMs = Date.now() - lastCaptureMs;
    receivedFrames++;
    stageEmpty.hidden = true;
    stageLabel.textContent = 'GENERATED LIVE';
    setStatus('Streaming generated frames.');
  } finally {
    bitmap.close();
  }
}

async function startMirror(): Promise<void> {
  if (stream || socket) return;
  const url = endpoint.value.trim();
  if (!/^wss?:\/\//i.test(url)) { setStatus('Enter a ws:// or wss:// model endpoint.', true); return; }
  const currentRun = ++runId;
  mode = MODES[modeSelect.value as keyof typeof MODES] ?? MODES.b200;
  capture.width = output.width = mode.width;
  capture.height = output.height = mode.height;
  stageResolution.textContent = `${mode.width} × ${mode.height}`;
  startButton.disabled = true;
  endpoint.disabled = true;
  modeSelect.disabled = true;
  prompt.disabled = true;
  setStatus('Requesting camera access…');
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' } });
    if (currentRun !== runId) return;
    video.srcObject = stream;
    await video.play();
    if (currentRun !== runId) return;
    const currentSocket = new WebSocket(url);
    currentSocket.binaryType = 'arraybuffer';
    socket = currentSocket;
    stopButton.disabled = false;
    setStatus('Connecting to the generative model…');
    currentSocket.onopen = () => { if (currentRun === runId) setStatus('Waiting for a model session…'); };
    currentSocket.onmessage = event => {
      if (currentRun !== runId || currentSocket !== socket) return;
      if (typeof event.data === 'string') {
        let message: Record<string, unknown>;
        try { message = JSON.parse(event.data) as Record<string, unknown>; }
        catch { setStatus('The model sent an invalid response.', true); return; }
        if (message.type === 'session_granted') {
          currentSocket.send(JSON.stringify(createStartMessage(prompt.value.trim(), mode.width, mode.height, mode.fps)));
        } else if (message.type === 'queue_position') {
          setStatus(`Waiting for model session · ${Number(message.ahead ?? message.position ?? 0)} ahead`);
        } else if (message.type === 'started') {
          setStatus('Model started. Waiting for generated frames…');
          let lastCount = 0;
          sendTimer = window.setInterval(() => { void sendFrame(currentRun); }, 1000 / mode.fps);
          metricsTimer = window.setInterval(() => {
            const fps = receivedFrames - lastCount;
            lastCount = receivedFrames;
            metrics.textContent = `OUTPUT ${fps} FPS · DELAY ${lastLatencyMs === null ? '—' : `${lastLatencyMs} MS`}`;
          }, 1000);
        } else if (message.type === 'output_frame') {
          pendingOutputMeta = message as OutputMetadata;
        } else if (message.type === 'chunk_done') {
          currentSocket.send(JSON.stringify({ type: 'ack', recv: receivedPackets }));
        } else if (message.type === 'error') {
          setStatus(`Model error: ${String(message.message ?? 'unknown error')}`, true);
          clearTimers();
        }
      } else if (event.data instanceof ArrayBuffer) {
        receivedPackets++;
        const meta = pendingOutputMeta;
        pendingOutputMeta = null;
        void displayGeneratedFrame(event.data, meta, currentRun).catch(error => setStatus(`Could not decode a generated frame: ${String(error)}`, true));
      }
    };
    currentSocket.onerror = () => { if (currentRun === runId) setStatus('Could not connect to the model endpoint.', true); };
    currentSocket.onclose = () => { if (currentRun === runId) stopMirror('Model connection closed.'); };
  } catch (error) {
    stopMirror('Camera or model connection failed.');
    setStatus(`Could not start: ${String(error)}`, true);
  }
}

startButton.addEventListener('click', () => { void startMirror(); });
stopButton.addEventListener('click', () => stopMirror());
window.addEventListener('beforeunload', () => stream?.getTracks().forEach(track => track.stop()));
