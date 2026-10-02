import type { RecordedFrameMeta } from './xr-recording';
import { FrameRecorder, PersonMasker, READ_WIDTH, Samples, fmt } from './xr-shared';

// Front (selfie) camera check. ARCore tracks motion and estimates depth only with the world-facing
// camera, so this mode runs outside WebXR: getUserMedia video, the person mask, and device rotation
// from the phone's motion sensors in place of a tracked pose.

type Orientation = NonNullable<RecordedFrameMeta['orientation']>;

class OrientationTracker {
  source: 'orientation sensor' | 'deviceorientation' | 'none' = 'none';
  private sensor: RelativeOrientationSensor | null = null;
  private euler: number[] | null = null;
  private readonly onEvent = (event: DeviceOrientationEvent): void => {
    if (event.alpha !== null && event.beta !== null && event.gamma !== null) this.euler = [event.alpha, event.beta, event.gamma];
  };

  start(): void {
    if (typeof RelativeOrientationSensor === 'undefined') { this.useEvents(); return; }
    try {
      const sensor = new RelativeOrientationSensor({ frequency: 60, referenceFrame: 'device' });
      sensor.onerror = event => { console.warn('Orientation sensor failed', event.error); this.sensor?.stop(); this.sensor = null; this.useEvents(); };
      sensor.start();
      this.sensor = sensor;
      this.source = 'orientation sensor';
    } catch (error) {
      console.warn('Orientation sensor unavailable', error);
      this.useEvents();
    }
  }

  stop(): void {
    this.sensor?.stop();
    this.sensor = null;
    window.removeEventListener('deviceorientation', this.onEvent);
  }

  read(): Orientation | null {
    const screenAngle = screen.orientation?.angle ?? 0;
    if (this.sensor?.quaternion) return { kind: 'quaternion', value: Array.from(this.sensor.quaternion), screenAngle };
    if (this.euler) return { kind: 'euler-deg', value: [...this.euler], screenAngle };
    return null;
  }

  private useEvents(): void {
    if (this.source === 'deviceorientation') return;
    window.addEventListener('deviceorientation', this.onEvent);
    this.source = 'deviceorientation';
  }
}

export class FrontCameraCheck {
  private readonly video = document.createElement('video');
  private readonly maskCanvas = document.createElement('canvas');
  private readonly maskContext = this.maskCanvas.getContext('2d')!;
  private readonly readCanvas = document.createElement('canvas');
  private readonly readContext = this.readCanvas.getContext('2d', { willReadFrequently: true })!;
  private readonly masker = new PersonMasker();
  private readonly recorder = new FrameRecorder();
  private readonly orientation = new OrientationTracker();
  private readonly frameMs = new Samples();
  private showMask = true;
  private running = true;
  private frames = 0;
  private lastFrameAt = 0;
  private lastHudAt = 0;

  private constructor(private readonly stream: MediaStream, private readonly stage: HTMLElement, private readonly hud: HTMLElement) {}

  static async start(stage: HTMLElement, hud: HTMLElement): Promise<FrontCameraCheck> {
    const video = { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } };
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { ...video, facingMode: { exact: 'user' } } });
    } catch (error) {
      // Laptop webcams often report no facing mode; fall back to any camera there.
      if (!(error instanceof Error && error.name === 'OverconstrainedError')) throw error;
      stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { ...video, facingMode: 'user' } });
    }
    const check = new FrontCameraCheck(stream, stage, hud);
    await check.begin();
    return check;
  }

  private async begin(): Promise<void> {
    this.video.muted = true;
    this.video.playsInline = true;
    this.video.srcObject = this.stream;
    this.stage.replaceChildren(this.video, this.maskCanvas);
    this.stage.hidden = false;
    await this.video.play();
    this.orientation.start();
    this.video.requestVideoFrameCallback(this.onFrame);
  }

  private readonly onFrame = (now: number): void => {
    if (!this.running) return;
    this.video.requestVideoFrameCallback(this.onFrame);
    if (this.lastFrameAt) this.frameMs.add(now - this.lastFrameAt);
    this.lastFrameAt = now;
    this.frames++;
    const recordDue = this.recorder.due(now);
    if (this.masker.segmenter || recordDue) {
      this.readFrame();
      const mask = this.masker.segment(this.readCanvas, now);
      if (mask && this.showMask) this.drawMask(mask);
      if (recordDue) {
        this.recorder.capture(now, this.readCanvas, {
          source: 'front-camera', cameraToWorld: null, projection: null, orientation: this.orientation.read(), depth: null,
        });
      }
    }
    if (!this.showMask || !this.masker.segmenter) this.maskContext.clearRect(0, 0, this.maskCanvas.width, this.maskCanvas.height);
    this.updateHud(now);
  };

  /** Unmirrored camera frame, READ_WIDTH wide; the on-screen video and mask are mirrored with CSS. */
  private readFrame(): void {
    const width = READ_WIDTH;
    const height = Math.round(READ_WIDTH * this.video.videoHeight / this.video.videoWidth);
    if (this.readCanvas.width !== width || this.readCanvas.height !== height) {
      this.readCanvas.width = width;
      this.readCanvas.height = height;
    }
    this.readContext.drawImage(this.video, 0, 0, width, height);
  }

  private drawMask(mask: { bytes: Uint8Array; width: number; height: number }): void {
    if (this.maskCanvas.width !== mask.width || this.maskCanvas.height !== mask.height) {
      this.maskCanvas.width = mask.width;
      this.maskCanvas.height = mask.height;
    }
    const image = this.maskContext.createImageData(mask.width, mask.height);
    for (let i = 0; i < mask.bytes.length; i++) {
      const offset = i * 4;
      image.data[offset] = 255;
      image.data[offset + 1] = 38;
      image.data[offset + 2] = 77;
      image.data[offset + 3] = mask.bytes[i] * 0.6;
    }
    this.maskContext.putImageData(image, 0, 0);
  }

  private updateHud(now: number): void {
    if (now - this.lastHudAt < 500) return;
    this.lastHudAt = now;
    const settings = this.stream.getVideoTracks()[0]?.getSettings();
    const frameMs = this.frameMs.median();
    const rotation = this.orientation.read();
    this.hud.textContent = [
      `FRONT CAMERA  ·  FPS ${fmt(frameMs && 1000 / frameMs, 1)}  ·  no AR tracking or depth`,
      `camera ${settings ? `${settings.width}×${settings.height} @ ${fmt(settings.frameRate ?? null, 0)} fps` : '—'}`,
      `mask ${this.masker.label}`,
      `rotation ${this.orientation.source}${rotation ? ` [${rotation.value.map(value => value.toFixed(2)).join(', ')}]` : ' (no reading yet)'}`,
      this.recorder.active ? this.recorder.label : 'Record saves frames with device rotation.',
    ].join('\n');
  }

  cycleView(): string {
    this.showMask = !this.showMask;
    return this.showMask ? 'View: mask' : 'View: camera';
  }

  async toggleMask(): Promise<string> {
    if (this.masker.segmenter) { this.masker.close(); return 'Mask: off'; }
    await this.masker.load();
    if (!this.running) { this.masker.close(); return 'Mask: off'; }
    return this.masker.segmenter ? `Mask: ${this.masker.delegate}` : 'Mask: failed';
  }

  toggleRecord(): string { return this.recorder.toggle() ? '■ Stop' : '● Record'; }

  stop(): object {
    this.running = false;
    const cameraSettings = this.stream.getVideoTracks()[0]?.getSettings() ?? null;
    this.masker.close();
    this.orientation.stop();
    this.stream.getTracks().forEach(track => track.stop());
    this.stage.replaceChildren();
    this.stage.hidden = true;
    const frameMs = this.frameMs.median();
    return {
      mode: 'front-camera',
      finishedAt: new Date().toISOString(),
      userAgent: navigator.userAgent,
      cameraSettings,
      frames: this.frames,
      medianFps: frameMs ? Number((1000 / frameMs).toFixed(1)) : null,
      readbackSize: this.readCanvas.width ? { width: this.readCanvas.width, height: this.readCanvas.height } : null,
      segmenter: this.masker.delegate,
      medianSegmentMs: this.masker.ms.median(),
      orientationSource: this.orientation.source,
      recording: this.recorder.summary,
    };
  }
}
