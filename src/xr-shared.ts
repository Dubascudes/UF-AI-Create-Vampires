import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';
import { encodeFramePacket, type RecordedFrameMeta } from './xr-recording';

export const READ_WIDTH = 480;
const RECORD_INTERVAL_MS = 200;
const MAX_UPLOADS_IN_FLIGHT = 6;

export class Samples {
  private values: number[] = [];
  constructor(private readonly size = 90) {}
  add(value: number): void { this.values.push(value); if (this.values.length > this.size) this.values.shift(); }
  median(): number | null {
    if (!this.values.length) return null;
    const sorted = [...this.values].sort((a, b) => a - b);
    return sorted[sorted.length >> 1];
  }
}

export const fmt = (value: number | null, digits = 0, unit = ''): string => value === null ? '—' : `${value.toFixed(digits)}${unit}`;

/** MediaPipe Selfie Segmenter, GPU delegate with CPU fallback. */
export class PersonMasker {
  segmenter: ImageSegmenter | null = null;
  delegate = '—';
  loading = false;
  readonly ms = new Samples();

  get label(): string {
    return this.segmenter ? `${fmt(this.ms.median(), 1, ' ms')} (${this.delegate})` : this.loading ? 'loading…' : 'off';
  }

  async load(): Promise<void> {
    if (this.segmenter || this.loading) return;
    this.loading = true;
    try {
      const vision = await FilesetResolver.forVisionTasks('/wasm');
      for (const delegate of ['GPU', 'CPU'] as const) {
        try {
          this.segmenter = await ImageSegmenter.createFromOptions(vision, {
            baseOptions: { modelAssetPath: '/models/selfie_segmenter.tflite', delegate },
            runningMode: 'VIDEO', outputCategoryMask: false, outputConfidenceMasks: true,
          });
          this.delegate = delegate;
          return;
        } catch (error) {
          console.warn(`MediaPipe ${delegate} delegate failed`, error);
        }
      }
      this.delegate = 'failed';
    } finally {
      this.loading = false;
    }
  }

  close(): void {
    this.segmenter?.close();
    this.segmenter = null;
  }

  /** Person confidence as bytes (0–255), rows top-down, or null when no mask is available. */
  segment(image: HTMLCanvasElement, time: number): { bytes: Uint8Array; width: number; height: number } | null {
    if (!this.segmenter) return null;
    const start = performance.now();
    const result = this.segmenter.segmentForVideo(image, time);
    try {
      const mask = result.confidenceMasks?.[0];
      if (!mask) return null;
      const values = mask.getAsFloat32Array();
      const bytes = new Uint8Array(values.length);
      for (let i = 0; i < values.length; i++) bytes[i] = values[i] * 255;
      return { bytes, width: mask.width, height: mask.height };
    } finally {
      result.close();
      this.ms.add(performance.now() - start);
    }
  }
}

/** Uploads about five frames per second to the dev server while recording. */
export class FrameRecorder {
  id: string | null = null;
  lastId: string | null = null;
  frames = 0;
  dropped = 0;
  failed = 0;
  private inFlight = 0;
  private lastAt = 0;

  get active(): boolean { return this.id !== null; }

  get label(): string {
    return `REC ${this.id}  ·  ${this.frames} frames  ·  dropped ${this.dropped}  ·  failed ${this.failed}`;
  }

  get summary(): { id: string | null; frames: number; dropped: number; failed: number } | null {
    return this.frames ? { id: this.lastId, frames: this.frames, dropped: this.dropped, failed: this.failed } : null;
  }

  toggle(): boolean {
    if (this.id) { this.id = null; return false; }
    this.id = this.lastId = new Date().toISOString().replace(/[:.]/g, '-').replace('Z', '');
    this.frames = this.dropped = this.failed = 0;
    this.lastAt = 0;
    return true;
  }

  due(time: number): boolean { return !!this.id && time - this.lastAt >= RECORD_INTERVAL_MS; }

  /** Encode the canvas as JPEG and upload it with metadata; depth must already be copied out of the XR frame. */
  capture(time: number, image: HTMLCanvasElement, meta: Omit<RecordedFrameMeta, 'index' | 'timeMs' | 'image'>, depth?: Uint8Array): void {
    if (!this.id) return;
    this.lastAt = time;
    if (this.inFlight >= MAX_UPLOADS_IN_FLIGHT) { this.dropped++; return; }
    const full: RecordedFrameMeta = { ...meta, index: this.frames++, timeMs: time, image: { width: image.width, height: image.height } };
    const id = this.id;
    this.inFlight++;
    image.toBlob(async blob => {
      try {
        if (!blob) throw new Error('JPEG encoding failed');
        const packet = encodeFramePacket(full, new Uint8Array(await blob.arrayBuffer()), depth);
        const response = await fetch(`/xr-record/${id}`, { method: 'POST', body: packet });
        if (!response.ok) throw new Error(await response.text());
      } catch (error) {
        this.failed++;
        console.error('Recording upload failed', error);
      } finally {
        this.inFlight--;
      }
    }, 'image/jpeg', 0.88);
  }
}

export function saveReport(summary: object): void {
  console.log('Phone check summary', summary);
  void fetch('/xr-report', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(summary) })
    .catch(error => console.warn('Could not save the report on the dev machine', error));
}
