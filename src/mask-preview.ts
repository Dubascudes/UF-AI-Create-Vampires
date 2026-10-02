import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';
import { fetchYoloPersonMask, type PersonMask } from './person-segmentation';
import './style.css';

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <main class="shell generated-shell">
    <header class="topbar"><div class="brand"><span class="brand-mark" aria-hidden="true">◌</span><span>ABSENT<span class="brand-separator">/</span> PERSON MASK COMPARISON</span></div><a href="/" class="preview-link">Back to mirror ↗</a></header>
    <section class="hero"><div class="hero-copy"><p class="eyebrow">LOCAL CAMERA TEST</p><h1>See exactly what<br />the model <em>masks.</em></h1><p class="intro">Green pixels are identified as a person. Move your hands and body through the view, then switch models to compare coverage and speed.</p><div class="hero-actions"><button id="cameraButton" class="button button-primary" type="button">Start camera <span aria-hidden="true">↗</span></button><span class="privacy-note"><span class="privacy-dot"></span> YOLO frames stay on this computer</span></div></div></section>
    <section class="workspace" aria-label="Person mask comparison"><div class="stage-panel"><div class="stage-topline"><span><span class="live-indicator"></span> <span id="stageLabel">CAMERA OFFLINE</span></span><span>416 × 240</span></div><div class="stage preview-stage"><video id="camera" autoplay muted playsinline aria-hidden="true"></video><canvas id="output" width="416" height="240" aria-label="Webcam image with person mask in green"></canvas><div class="stage-empty" id="stageEmpty"><div class="empty-symbol">◌</div><p>Compare person masks</p><span>Start the camera to see mask coverage</span></div></div><div class="stage-footer"><span id="status" role="status" aria-live="polite">Ready.</span><span id="metrics">MASK — · WALL — · COVERAGE —</span></div></div><aside class="control-panel"><div class="panel-heading"><span>MASK CONTROLS</span><span class="panel-icon">✳</span></div><div class="object-control"><h2>Model</h2><p>Switch while the camera is active. YOLO uses the local RTX GPU; MediaPipe runs in the browser on CPU.</p><label class="field-label" for="modelSelect">Person segmenter</label><select class="field-input" id="modelSelect"><option value="yolo">YOLO26n-seg · local GPU</option><option value="mediapipe">MediaPipe Selfie Segmenter · CPU</option></select></div><div class="object-control"><h2>What to check</h2><p>Raise each hand, extend your arms, turn sideways, and move partly out of view. Look for uncovered skin or large green areas that belong to the room.</p></div><button id="stopButton" class="text-button muted" type="button" disabled>Stop camera</button></aside></section>
  </main>`;

const select = <T extends Element>(query: string): T => document.querySelector<T>(query)!;
const video = select<HTMLVideoElement>('#camera');
const output = select<HTMLCanvasElement>('#output');
const outputContext = output.getContext('2d')!;
const capture = document.createElement('canvas');
capture.width = output.width;
capture.height = output.height;
const captureContext = capture.getContext('2d')!;
const cameraButton = select<HTMLButtonElement>('#cameraButton');
const stopButton = select<HTMLButtonElement>('#stopButton');
const modelSelect = select<HTMLSelectElement>('#modelSelect');
const status = select<HTMLElement>('#status');
const metrics = select<HTMLElement>('#metrics');
const stageLabel = select<HTMLElement>('#stageLabel');
const stageEmpty = select<HTMLElement>('#stageEmpty');
let stream: MediaStream | null = null;
let mediaPipe: ImageSegmenter | null = null;
let mediaPipeLoading: Promise<ImageSegmenter> | null = null;
let timer = 0;
let generation = 0;

function setStatus(message: string, error = false): void {
  status.textContent = message;
  status.classList.toggle('error', error);
}

async function loadMediaPipe(): Promise<ImageSegmenter> {
  if (mediaPipe) return mediaPipe;
  if (!mediaPipeLoading) mediaPipeLoading = (async () => {
    const vision = await FilesetResolver.forVisionTasks('/wasm');
    return ImageSegmenter.createFromOptions(vision, {
      baseOptions: { modelAssetPath: '/models/selfie_segmenter.tflite', delegate: 'CPU' },
      runningMode: 'VIDEO', outputCategoryMask: false, outputConfidenceMasks: true,
    });
  })();
  try { mediaPipe = await mediaPipeLoading; return mediaPipe; }
  catch (error) { mediaPipeLoading = null; throw error; }
}

function drawCamera(): void {
  const sw = video.videoWidth;
  const sh = video.videoHeight;
  const aspect = capture.width / capture.height;
  if (sw / sh > aspect) {
    const width = sh * aspect;
    captureContext.drawImage(video, (sw - width) / 2, 0, width, sh, 0, 0, capture.width, capture.height);
  } else {
    const height = sw / aspect;
    captureContext.drawImage(video, 0, (sh - height) / 2, sw, height, 0, 0, capture.width, capture.height);
  }
}

async function detect(model: string): Promise<PersonMask> {
  if (model === 'yolo') return fetchYoloPersonMask(capture);
  const segmenter = await loadMediaPipe();
  const start = performance.now();
  const result = segmenter.segmentForVideo(capture, start);
  const mask = result.confidenceMasks?.[1] ?? result.confidenceMasks?.[0];
  if (!mask) throw new Error('MediaPipe returned no person mask');
  return { values: mask.getAsFloat32Array().slice(), width: mask.width, height: mask.height,
    inferenceMs: performance.now() - start };
}

function drawOverlay(mask: PersonMask): number {
  outputContext.drawImage(capture, 0, 0);
  const pixels = outputContext.getImageData(0, 0, output.width, output.height);
  let covered = 0;
  for (let y = 0; y < output.height; y++) {
    const my = Math.min(mask.height - 1, Math.floor(y * mask.height / output.height));
    for (let x = 0; x < output.width; x++) {
      const mx = Math.min(mask.width - 1, Math.floor(x * mask.width / output.width));
      const confidence = mask.values[my * mask.width + mx];
      if (confidence < 0.35) continue;
      covered++;
      const offset = (y * output.width + x) * 4;
      pixels.data[offset] = pixels.data[offset] * 0.45 + 40 * 0.55;
      pixels.data[offset + 1] = pixels.data[offset + 1] * 0.45 + 230 * 0.55;
      pixels.data[offset + 2] = pixels.data[offset + 2] * 0.45 + 70 * 0.55;
    }
  }
  outputContext.putImageData(pixels, 0, 0);
  return covered / (output.width * output.height);
}

async function step(currentGeneration: number): Promise<void> {
  if (currentGeneration !== generation || !stream || video.readyState < 2) return;
  const start = performance.now();
  const model = modelSelect.value;
  try {
    drawCamera();
    const mask = await detect(model);
    if (currentGeneration !== generation || modelSelect.value !== model) return;
    const coverage = drawOverlay(mask);
    stageEmpty.hidden = true;
    stageLabel.textContent = model === 'yolo' ? 'YOLO MASK LIVE' : 'MEDIAPIPE MASK LIVE';
    metrics.textContent = `MASK ${Math.round(mask.inferenceMs)} MS · WALL ${Math.round(performance.now() - start)} MS · COVERAGE ${Math.round(coverage * 100)}%`;
    setStatus('Green shows detected person pixels.');
  } catch (error) {
    setStatus(`Mask failed: ${String(error)}`, true);
  } finally {
    if (currentGeneration === generation && stream) timer = window.setTimeout(() => { void step(currentGeneration); }, Math.max(0, 33 - (performance.now() - start)));
  }
}

function stop(): void {
  generation++;
  window.clearTimeout(timer);
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  outputContext.clearRect(0, 0, output.width, output.height);
  stageEmpty.hidden = false;
  stageLabel.textContent = 'CAMERA OFFLINE';
  metrics.textContent = 'MASK — · WALL — · COVERAGE —';
  cameraButton.disabled = false;
  stopButton.disabled = true;
  setStatus('Camera stopped.');
}

cameraButton.addEventListener('click', async () => {
  if (stream) return;
  const currentGeneration = ++generation;
  cameraButton.disabled = true;
  setStatus('Requesting camera access…');
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
    if (currentGeneration !== generation) return;
    video.srcObject = stream;
    await video.play();
    stopButton.disabled = false;
    void step(currentGeneration);
  } catch (error) {
    stop();
    setStatus(`Could not start camera: ${String(error)}`, true);
  }
});
stopButton.addEventListener('click', stop);
window.addEventListener('beforeunload', () => { stream?.getTracks().forEach(track => track.stop()); mediaPipe?.close(); });
