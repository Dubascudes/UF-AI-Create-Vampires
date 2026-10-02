import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';
import { BackgroundModel, type BackgroundResult } from './background';
import { liveOpacity, personCoverage } from './mask';
import { ObjectTracker } from './object-tracker';
import './style.css';

const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `
  <main class="shell">
    <header class="topbar">
      <div class="brand"><span class="brand-mark" aria-hidden="true">◌</span><span>ABSENT<span class="brand-separator">/</span> A LIVE VANISHING MIRROR</span></div>
      <span class="edition">GATOR CREATE · 2026</span>
    </header>

    <section class="hero" aria-labelledby="title">
      <div class="hero-copy">
        <p class="eyebrow">AN EXPERIMENT IN PRESENCE</p>
        <h1 id="title">What does a room<br />look like <em>without us?</em></h1>
        <p class="intro">A live mirror that makes people disappear from a familiar space. The room is learned automatically as it comes into view.</p>
        <div class="hero-actions">
          <button id="cameraButton" class="button button-primary" type="button">Start camera <span aria-hidden="true">↗</span></button>
          <span class="privacy-note"><span class="privacy-dot"></span> Video is processed on this device</span>
        </div>
      </div>
      <div class="hero-index" aria-hidden="true"><span>01 / 03</span><span>THE OBSERVATION</span></div>
    </section>

    <section class="workspace" aria-label="Live mirror workspace">
      <div class="stage-panel">
        <div class="stage-topline"><span><span class="live-indicator"></span> <span id="stageLabel">CAMERA OFFLINE</span></span><span id="frameLabel">NO SIGNAL</span></div>
        <div class="stage" id="stage">
          <video id="video" autoplay muted playsinline aria-hidden="true"></video>
          <canvas id="output" aria-label="Processed webcam view"></canvas>
          <div class="stage-empty" id="stageEmpty"><div class="empty-symbol">◌</div><p>Your room appears here</p><span>Connect a camera to begin</span></div>
        </div>
        <div class="stage-footer"><span id="status" role="status" aria-live="polite">Ready when you are.</span><span id="coverage">ROOM LEARNED —</span></div>
      </div>

      <aside class="control-panel" aria-label="Mirror controls">
        <div class="panel-heading"><span>CONTROL ROOM</span><span class="panel-icon">✳</span></div>
        <div class="step"><span class="step-num">01</span><div><h2>Frame the scene</h2><p>Place the camera somewhere steady and keep the view fixed as the room is learned.</p></div></div>
        <div class="step"><span class="step-num">02</span><div><h2>Reveal the room</h2><p>Move naturally. The mirror learns each part of the background when no person covers it.</p></div></div>
        <div class="step"><span class="step-num">03</span><div><h2>Watch yourself vanish</h2><p>The learned room replaces people live. Areas never seen by the camera use an approximate fill until revealed.</p></div></div>
        <div class="object-control"><h2>Generative room fill</h2><p id="inpaintStatus" role="status" aria-live="polite">Starts with the camera when the local inpainter is available.</p></div>
        <div class="object-control"><h2>Make an object float</h2><p>Choose an object, then click its visible surface in the mirror. Keep it facing the camera as you move.</p><button id="objectButton" class="button button-secondary" type="button" disabled>Choose object <span aria-hidden="true">↗</span></button><button id="clearObjectButton" class="text-button muted" type="button" hidden>Clear selected object</button><p id="objectStatus" role="status" aria-live="polite">Object selection loads with the camera.</p></div>
        <div class="divider"></div>
        <div class="setting"><label for="sensitivity">Person sensitivity <output id="sensitivityValue">50%</output></label><input id="sensitivity" type="range" min="25" max="75" value="50" /><div class="scale"><span>MORE REMOVAL</span><span>MORE DETAIL</span></div></div>
        <label class="toggle-row"><span>Show person mask</span><input id="showMask" type="checkbox" role="switch" /><span class="switch" aria-hidden="true"></span></label>
        <button id="fullscreenButton" class="text-button" type="button">Enter fullscreen <span aria-hidden="true">↗</span></button>
        <button id="resetButton" class="text-button muted" type="button" disabled>Reset learned room</button>
        <button id="stopButton" class="text-button muted" type="button" disabled>Stop camera</button>
      </aside>
    </section>

    <footer class="footer"><span>ABSENT <span class="footer-slash">/</span> MADE TO QUESTION WHAT WE CHOOSE TO SEE</span><span>CAMERA FRAMES ARE NEVER SAVED OR UPLOADED</span></footer>
  </main>`;

const video = document.querySelector<HTMLVideoElement>('#video')!;
const output = document.querySelector<HTMLCanvasElement>('#output')!;
const stage = document.querySelector<HTMLDivElement>('#stage')!;
const stageEmpty = document.querySelector<HTMLDivElement>('#stageEmpty')!;
const cameraButton = document.querySelector<HTMLButtonElement>('#cameraButton')!;
const resetButton = document.querySelector<HTMLButtonElement>('#resetButton')!;
const objectButton = document.querySelector<HTMLButtonElement>('#objectButton')!;
const clearObjectButton = document.querySelector<HTMLButtonElement>('#clearObjectButton')!;
const objectStatus = document.querySelector<HTMLElement>('#objectStatus')!;
const inpaintStatus = document.querySelector<HTMLElement>('#inpaintStatus')!;
const stopButton = document.querySelector<HTMLButtonElement>('#stopButton')!;
const fullscreenButton = document.querySelector<HTMLButtonElement>('#fullscreenButton')!;
const showMask = document.querySelector<HTMLInputElement>('#showMask')!;
const sensitivity = document.querySelector<HTMLInputElement>('#sensitivity')!;
const sensitivityValue = document.querySelector<HTMLOutputElement>('#sensitivityValue')!;
const status = document.querySelector<HTMLElement>('#status')!;
const coverageLabel = document.querySelector<HTMLElement>('#coverage')!;
const stageLabel = document.querySelector<HTMLElement>('#stageLabel')!;
const frameLabel = document.querySelector<HTMLElement>('#frameLabel')!;
const backgroundCanvas = document.createElement('canvas');
const cameraFrame = document.createElement('canvas');
const live = document.createElement('canvas');
const mask = document.createElement('canvas');
const maskOverlay = document.createElement('canvas');
const objectMaskCanvas = document.createElement('canvas');
const objectLayer = document.createElement('canvas');
const outputContext = output.getContext('2d')!;
const backgroundContext = backgroundCanvas.getContext('2d')!;
const cameraFrameContext = cameraFrame.getContext('2d', { willReadFrequently: true })!;
const liveContext = live.getContext('2d', { willReadFrequently: true })!;
const maskContext = mask.getContext('2d')!;
const maskOverlayContext = maskOverlay.getContext('2d')!;
const objectMaskContext = objectMaskCanvas.getContext('2d')!;
const objectLayerContext = objectLayer.getContext('2d')!;

let stream: MediaStream | null = null;
let segmenter: ImageSegmenter | null = null;
let background: BackgroundModel | null = null;
let lastInference = 0;
let animationFrame = 0;
let cameraGeneration = 0;
let objectWorker: Worker | null = null;
let objectReady = false;
let selectingObject = false;
let objectTracker: ObjectTracker | null = null;
let selectionId = 0;
let selectionFrame: Uint8ClampedArray | null = null;
let lostFrames = 0;
let sceneGeneration = 0;
let inpaintRequestId = 0;
let inpaintPending = false;
let lastInpaintRequest = -5000;

function setStatus(message: string, error = false): void {
  status.textContent = message;
  status.classList.toggle('error', error);
}

function resizeCanvases(): void {
  const width = Math.min(video.videoWidth || 640, 640);
  const height = Math.round(width * (video.videoHeight || 540) / (video.videoWidth || 960));
  for (const canvas of [output, backgroundCanvas, cameraFrame, live, objectMaskCanvas, objectLayer]) {
    canvas.width = width;
    canvas.height = height;
  }
  background = new BackgroundModel(width, height);
  stage.style.aspectRatio = `${width} / ${height}`;
  frameLabel.textContent = `${width} × ${height}`;
}

function clearSelectedObject(): void {
  objectTracker = null;
  selectionFrame = null;
  lostFrames = 0;
  objectMaskContext.clearRect(0, 0, objectMaskCanvas.width, objectMaskCanvas.height);
  clearObjectButton.hidden = true;
  objectButton.innerHTML = 'Choose object <span aria-hidden="true">↗</span>';
}

function installObjectMask(values: Float32Array, width: number, height: number): void {
  const resized = new Float32Array(output.width * output.height);
  for (let y = 0; y < output.height; y++) {
    const sourceY = Math.min(height - 1, Math.floor((y + 0.5) * height / output.height));
    for (let x = 0; x < output.width; x++) {
      const sourceX = Math.min(width - 1, Math.floor((x + 0.5) * width / output.width));
      resized[y * output.width + x] = values[sourceY * width + sourceX];
    }
  }
  const tracker = ObjectTracker.create(selectionFrame!, resized, output.width, output.height);
  if (!tracker) {
    objectStatus.textContent = 'Could not isolate that object. Choose a visible part and try again.';
    objectButton.disabled = false;
    selectionFrame = null;
    return;
  }
  objectTracker = tracker;
  background?.forget(resized, output.width, output.height);
  lostFrames = 0;
  const pixels = objectMaskContext.createImageData(output.width, output.height);
  for (let i = 0; i < resized.length; i++) {
    const offset = i * 4;
    pixels.data[offset] = pixels.data[offset + 1] = pixels.data[offset + 2] = 255;
    pixels.data[offset + 3] = Math.round(Math.max(0, Math.min(1, (resized[i] - 0.25) / 0.5)) * 255);
  }
  objectMaskContext.putImageData(pixels, 0, 0);
  objectStatus.textContent = 'Object kept visible. Keep it facing the camera.';
  objectButton.disabled = false;
  objectButton.innerHTML = 'Choose another object <span aria-hidden="true">↗</span>';
  clearObjectButton.hidden = false;
  selectionFrame = null;
}

function startObjectWorker(): void {
  objectReady = false;
  objectButton.disabled = true;
  objectStatus.textContent = 'Loading object selection…';
  const worker = new Worker('/object-worker.js');
  objectWorker = worker;
  worker.onmessage = (event: MessageEvent) => {
    if (worker !== objectWorker) return;
    const message = event.data as { type: string; id?: number; width?: number; height?: number; values?: Float32Array; message?: string };
    if (message.type === 'ready') {
      objectReady = true;
      objectButton.disabled = false;
      objectStatus.textContent = 'Ready. Choose an object, then click it in the mirror.';
    } else if (message.type === 'selected' && message.id === selectionId && message.values && message.width && message.height && selectionFrame) {
      installObjectMask(message.values, message.width, message.height);
    } else if (message.type === 'error') {
      if (message.id !== undefined && message.id !== selectionId) return;
      selectionFrame = null;
      objectButton.disabled = !objectReady;
      objectStatus.textContent = `Object selection failed: ${message.message ?? 'Unknown error'}`;
    }
  };
  worker.onerror = () => {
    objectReady = false;
    objectButton.disabled = true;
    objectStatus.textContent = 'Object selection could not start.';
  };
  worker.postMessage({ type: 'init' });
}

function requestInpaint(learned: BackgroundResult, model: BackgroundModel): void {
  const now = performance.now();
  if (inpaintPending || learned.knownFraction < 0.15 || learned.missingFraction < 0.005 || now - lastInpaintRequest < 5000) return;
  const id = ++inpaintRequestId;
  const generation = sceneGeneration;
  inpaintPending = true;
  lastInpaintRequest = now;
  inpaintStatus.textContent = 'Generating the unseen part of the room…';
  const missing = learned.missingMask;
  const maskCanvas = document.createElement('canvas');
  maskCanvas.width = model.width;
  maskCanvas.height = model.height;
  const maskContext = maskCanvas.getContext('2d')!;
  const maskPixels = maskContext.createImageData(model.width, model.height);
  for (let i = 0; i < missing.length; i++) {
    const offset = i * 4;
    maskPixels.data[offset] = missing[i];
    maskPixels.data[offset + 1] = missing[i];
    maskPixels.data[offset + 2] = missing[i];
    maskPixels.data[offset + 3] = 255;
  }
  maskContext.putImageData(maskPixels, 0, 0);
  void fetch('/api/inpaint', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: backgroundCanvas.toDataURL('image/png'), mask: maskCanvas.toDataURL('image/png') }),
  }).then(async response => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const elapsed = response.headers.get('X-Inference-Ms');
    const bitmap = await createImageBitmap(await response.blob());
    try {
      if (id !== inpaintRequestId || generation !== sceneGeneration || background !== model) return;
      maskContext.clearRect(0, 0, model.width, model.height);
      maskContext.drawImage(bitmap, 0, 0, model.width, model.height);
      model.installGenerated(maskContext.getImageData(0, 0, model.width, model.height).data, missing);
      inpaintStatus.textContent = `AI fill ready${elapsed ? ` · ${Math.round(Number(elapsed))} ms` : ''}. Real room pixels replace it when revealed.`;
    } finally {
      bitmap.close();
    }
  }).catch(error => {
    if (id === inpaintRequestId) inpaintStatus.textContent = `Local inpainter unavailable (${String(error)}). Approximate fill remains active.`;
  }).finally(() => {
    if (id === inpaintRequestId) inpaintPending = false;
  });
}

function drawFrame(): void {
  if (!stream || video.readyState < 2) return;
  const now = performance.now();
  if (segmenter && now - lastInference >= 33) {
    lastInference = now;
    try {
      cameraFrameContext.drawImage(video, 0, 0, cameraFrame.width, cameraFrame.height);
      segmenter.segmentForVideo(cameraFrame, now, (result) => {
        const personMask = result.confidenceMasks?.[1] ?? result.confidenceMasks?.[0];
        if (!personMask || !background) return;
        const values = personMask.getAsFloat32Array();
        const livePixels = cameraFrameContext.getImageData(0, 0, cameraFrame.width, cameraFrame.height);
        if (objectTracker && !objectTracker.update(livePixels.data)) {
          lostFrames++;
          if (lostFrames >= 5) {
            clearSelectedObject();
            objectStatus.textContent = 'Object tracking was lost. Choose the object again.';
          }
        } else {
          lostFrames = 0;
        }
        const preserved = objectTracker?.projectMask(personMask.width, personMask.height);
        const learned = background.observe(livePixels.data, values, personMask.width, personMask.height, preserved);
        backgroundContext.putImageData(new ImageData(learned.background, background.width, background.height), 0, 0);
        if (learned.knownFraction > 0.995 && inpaintStatus.textContent?.startsWith('Waiting')) {
          inpaintStatus.textContent = 'Room visible. No generated fill needed.';
        }
        requestInpaint(learned, background);
        coverageLabel.textContent = `ROOM LEARNED ${Math.round(learned.knownFraction * 100)}% · PERSON ${Math.round(personCoverage(values) * 100)}%`;
        stageLabel.textContent = 'VANISHING MIRROR';
        if (learned.knownFraction > 0.98) setStatus('Room learned. People are replaced live.');
        else setStatus('Learning the room. Move to reveal hidden areas.');
        mask.width = personMask.width;
        mask.height = personMask.height;
        maskOverlay.width = personMask.width;
        maskOverlay.height = personMask.height;
        const pixels = maskContext.createImageData(mask.width, mask.height);
        const overlayPixels = maskOverlayContext.createImageData(mask.width, mask.height);
        const threshold = Number(sensitivity.value) / 100;
        for (let i = 0; i < learned.mask.length; i++) {
          const offset = i * 4;
          pixels.data[offset] = 255;
          pixels.data[offset + 1] = 255;
          pixels.data[offset + 2] = 255;
          pixels.data[offset + 3] = liveOpacity(learned.mask[i], threshold);
          overlayPixels.data[offset] = 180;
          overlayPixels.data[offset + 1] = 255;
          overlayPixels.data[offset + 2] = 134;
          overlayPixels.data[offset + 3] = 255 - pixels.data[offset + 3];
        }
        maskContext.putImageData(pixels, 0, 0);
        maskOverlayContext.putImageData(overlayPixels, 0, 0);
      });
    } catch (error) {
      console.error(error);
      setStatus('Segmentation paused. Restart the camera to retry.', true);
      segmenter.close();
      segmenter = null;
    }
  }

  outputContext.clearRect(0, 0, output.width, output.height);
  if (!background || !segmenter || !mask.width) {
    outputContext.drawImage(video, 0, 0, output.width, output.height);
  } else {
    outputContext.drawImage(backgroundCanvas, 0, 0);
    liveContext.globalCompositeOperation = 'source-over';
    liveContext.clearRect(0, 0, live.width, live.height);
    liveContext.drawImage(cameraFrame, 0, 0, live.width, live.height);
    liveContext.globalCompositeOperation = 'destination-in';
    liveContext.drawImage(mask, 0, 0, live.width, live.height);
    liveContext.globalCompositeOperation = 'source-over';
    outputContext.drawImage(live, 0, 0);
  }
  if (showMask.checked && mask.width && segmenter) {
    outputContext.save();
    outputContext.globalAlpha = 0.32;
    outputContext.globalCompositeOperation = 'source-over';
    outputContext.drawImage(maskOverlay, 0, 0, output.width, output.height);
    outputContext.restore();
  }
  if (objectTracker && segmenter) {
    objectLayerContext.clearRect(0, 0, objectLayer.width, objectLayer.height);
    objectLayerContext.globalCompositeOperation = 'source-over';
    objectLayerContext.drawImage(cameraFrame, 0, 0, objectLayer.width, objectLayer.height);
    objectLayerContext.globalCompositeOperation = 'destination-in';
    objectLayerContext.drawImage(objectMaskCanvas, objectTracker.offsetX, objectTracker.offsetY);
    objectLayerContext.globalCompositeOperation = 'source-over';
    outputContext.drawImage(objectLayer, 0, 0);
  }
  animationFrame = requestAnimationFrame(drawFrame);
}

async function startCamera(): Promise<void> {
  if (stream) return;
  const generation = ++cameraGeneration;
  cameraButton.disabled = true;
  setStatus('Requesting camera access…');
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { width: { ideal: 640 }, height: { ideal: 360 }, facingMode: 'user' } });
    if (generation !== cameraGeneration) { stream.getTracks().forEach(track => track.stop()); stream = null; return; }
    video.srcObject = stream;
    await video.play();
    resizeCanvases();
    stageEmpty.hidden = true;
    stageLabel.textContent = 'LIVE CAMERA';
    resetButton.disabled = false;
    stopButton.disabled = false;
    cameraButton.textContent = 'Camera active';
    drawFrame();
    setStatus('Loading the person segmentation model…');
    const vision = await FilesetResolver.forVisionTasks('/wasm');
    const loaded = await ImageSegmenter.createFromOptions(vision, {
      baseOptions: { modelAssetPath: '/models/selfie_segmenter.tflite', delegate: 'CPU' },
      runningMode: 'VIDEO',
      outputCategoryMask: false,
      outputConfidenceMasks: true,
    });
    if (generation !== cameraGeneration) { loaded.close(); return; }
    segmenter = loaded;
    setStatus('Model ready. Learning the room automatically.');
    inpaintStatus.textContent = 'Waiting for enough of the room to become visible…';
    startObjectWorker();
  } catch (error) {
    console.error(error);
    const message = error instanceof DOMException && error.name === 'NotAllowedError'
      ? 'Camera permission denied. Allow access in your browser and retry.'
      : 'Camera or model failed to start. Check the setup instructions and retry.';
    setStatus(message, true);
    stopCamera();
  }
}

function stopCamera(): void {
  cameraGeneration++;
  cancelAnimationFrame(animationFrame);
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  segmenter?.close();
  segmenter = null;
  background = null;
  sceneGeneration++;
  inpaintRequestId++;
  inpaintPending = false;
  lastInpaintRequest = -5000;
  objectWorker?.terminate();
  objectWorker = null;
  objectReady = false;
  selectingObject = false;
  selectionId++;
  clearSelectedObject();
  stage.classList.remove('is-selecting');
  objectButton.disabled = true;
  objectStatus.textContent = 'Object selection loads with the camera.';
  inpaintStatus.textContent = 'Starts with the camera when the local inpainter is available.';
  mask.width = 0;
  maskOverlay.width = 0;
  stageEmpty.hidden = false;
  stageLabel.textContent = 'CAMERA OFFLINE';
  frameLabel.textContent = 'NO SIGNAL';
  coverageLabel.textContent = 'ROOM LEARNED —';
  resetButton.disabled = true;
  stopButton.disabled = true;
  cameraButton.disabled = false;
  cameraButton.innerHTML = 'Start camera <span aria-hidden="true">↗</span>';
}

cameraButton.addEventListener('click', startCamera);
objectButton.addEventListener('click', () => {
  if (!objectReady) return;
  selectingObject = !selectingObject;
  stage.classList.toggle('is-selecting', selectingObject);
  objectStatus.textContent = selectingObject
    ? 'Click an uncovered part of the object in the mirror.'
    : 'Object selection cancelled.';
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
  selectionFrame = cameraFrameContext.getImageData(0, 0, cameraFrame.width, cameraFrame.height).data;
  try {
    const bitmap = await createImageBitmap(cameraFrame);
    if (!objectWorker || id !== selectionId) { bitmap.close(); return; }
    objectWorker.postMessage({ type: 'select', id, bitmap, x, y }, [bitmap]);
  } catch (error) {
    objectButton.disabled = false;
    objectStatus.textContent = `Could not capture the selected frame: ${String(error)}`;
  }
});
clearObjectButton.addEventListener('click', () => {
  selectionId++;
  clearSelectedObject();
  objectStatus.textContent = 'Object cleared. Choose another when ready.';
});
resetButton.addEventListener('click', () => {
  background?.reset();
  sceneGeneration++;
  inpaintRequestId++;
  inpaintPending = false;
  lastInpaintRequest = -5000;
  inpaintStatus.textContent = 'Room reset. Generative fill will update.';
  coverageLabel.textContent = 'ROOM LEARNED 0%';
  setStatus('Room reset. Learning starts with the next frame.');
});
stopButton.addEventListener('click', () => { stopCamera(); setStatus('Camera stopped.'); });
fullscreenButton.addEventListener('click', async () => {
  if (document.fullscreenElement) await document.exitFullscreen();
  else await stage.requestFullscreen();
});
document.addEventListener('fullscreenchange', () => {
  fullscreenButton.innerHTML = document.fullscreenElement ? 'Exit fullscreen <span aria-hidden="true">↗</span>' : 'Enter fullscreen <span aria-hidden="true">↗</span>';
});
sensitivity.addEventListener('input', () => { sensitivityValue.value = `${sensitivity.value}%`; });
window.addEventListener('beforeunload', () => stream?.getTracks().forEach(track => track.stop()));
