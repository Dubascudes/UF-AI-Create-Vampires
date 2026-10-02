import { bytesToUnit } from './local-compose';

export interface PersonMask { values: Float32Array; width: number; height: number; inferenceMs: number }

export async function decodeYoloMaskResponse(response: Response): Promise<PersonMask> {
  if (!response.ok) throw new Error(`YOLO service returned HTTP ${response.status}: ${await response.text()}`);
  const width = Number(response.headers.get('X-Mask-Width'));
  const height = Number(response.headers.get('X-Mask-Height'));
  const inferenceMs = Number(response.headers.get('X-Inference-Ms'));
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 ||
      width * height !== bytes.length) throw new Error('YOLO mask dimensions do not match the response');
  const values = Float32Array.from(bytes, value => value / 255);
  return { values, width, height, inferenceMs };
}

export async function fetchYoloPersonMask(canvas: HTMLCanvasElement): Promise<PersonMask> {
  const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.82));
  if (!blob) throw new Error('Could not encode frame for YOLO');
  const response = await fetch('/seg/mask', { method: 'POST', headers: { 'Content-Type': 'image/jpeg' }, body: blob });
  return decodeYoloMaskResponse(response);
}

export function encodeYoloStreamFrame(requestId: number, frame: ImageData): ArrayBuffer {
  if (frame.width < 1 || frame.height < 1 || frame.width > 1280 || frame.height > 720 ||
      frame.data.length !== frame.width * frame.height * 4) throw new Error('Invalid RGBA frame dimensions');
  const result = new ArrayBuffer(8 + frame.data.length);
  const header = new DataView(result);
  header.setUint32(0, requestId, true);
  header.setUint16(4, frame.width, true);
  header.setUint16(6, frame.height, true);
  new Uint8Array(result, 8).set(frame.data);
  return result;
}

export function decodeYoloStreamResponse(data: ArrayBuffer): PersonMask & { requestId: number } {
  if (data.byteLength < 12) throw new Error('YOLO stream response is incomplete');
  const header = new DataView(data);
  const requestId = header.getUint32(0, true);
  const width = header.getUint16(4, true);
  const height = header.getUint16(6, true);
  const inferenceMs = header.getFloat32(8, true);
  if (width < 1 || height < 1 || data.byteLength !== 12 + width * height) {
    throw new Error('YOLO stream mask dimensions do not match the response');
  }
  return { requestId, width, height, inferenceMs,
    values: bytesToUnit(new Uint8Array(data, 12)) };
}

export class YoloStream {
  private readonly pending = new Map<number, { resolve: (mask: PersonMask) => void; reject: (error: Error) => void }>();
  private nextId = 1;

  private constructor(private readonly socket: WebSocket) {
    socket.binaryType = 'arraybuffer';
    socket.onmessage = event => {
      try {
        if (!(event.data instanceof ArrayBuffer)) throw new Error('YOLO stream returned a non-binary mask');
        const result = decodeYoloStreamResponse(event.data);
        const request = this.pending.get(result.requestId);
        if (!request) return;
        this.pending.delete(result.requestId);
        request.resolve(result);
      } catch (error) { this.rejectAll(new Error(String(error))); }
    };
    socket.onclose = () => this.rejectAll(new Error('Local YOLO stream closed'));
    socket.onerror = () => this.rejectAll(new Error('Local YOLO stream failed'));
  }

  static async connect(url = 'ws://127.0.0.1:8001/seg/stream'): Promise<YoloStream> {
    const socket = new WebSocket(url);
    return new Promise((resolve, reject) => {
      socket.onopen = () => resolve(new YoloStream(socket));
      socket.onerror = () => reject(new Error('Could not connect to the local YOLO stream'));
      socket.onclose = () => reject(new Error('Local YOLO stream closed before connecting'));
    });
  }

  segment(frame: ImageData): Promise<PersonMask> {
    if (this.socket.readyState !== WebSocket.OPEN) return Promise.reject(new Error('Local YOLO stream is unavailable'));
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      try { this.socket.send(encodeYoloStreamFrame(id, frame)); }
      catch (error) { this.pending.delete(id); reject(new Error(String(error))); }
    });
  }

  close(): void {
    this.rejectAll(new Error('Local YOLO stream stopped'));
    this.socket.close();
  }

  private rejectAll(error: Error): void {
    for (const request of this.pending.values()) request.reject(error);
    this.pending.clear();
  }
}
