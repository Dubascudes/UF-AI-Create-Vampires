export type DepthFormat = 'luminance-alpha' | 'unsigned-short' | 'float32';

export interface DepthStats { validFraction: number; min: number; median: number; max: number }

/** Convert a WebXR CPU depth buffer (row-major, no padding) to meters; invalid samples become 0. */
export function depthToMeters(
  data: ArrayBuffer, format: DepthFormat, rawValueToMeters: number, width: number, height: number,
): Float32Array {
  const count = width * height;
  const bytes = format === 'float32' ? 4 : 2;
  if (data.byteLength < count * bytes) throw new Error('Depth buffer is smaller than its dimensions');
  const raw = format === 'float32' ? new Float32Array(data, 0, count) : new Uint16Array(data, 0, count);
  const meters = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const value = raw[i] * rawValueToMeters;
    meters[i] = Number.isFinite(value) && value > 0 ? value : 0;
  }
  return meters;
}

export function depthStats(meters: Float32Array): DepthStats {
  const valid = meters.filter(value => value > 0).sort();
  if (!valid.length) return { validFraction: 0, min: 0, median: 0, max: 0 };
  return {
    validFraction: valid.length / meters.length,
    min: valid[0],
    median: valid[valid.length >> 1],
    max: valid[valid.length - 1],
  };
}

/** Near is warm, far is cool; invalid samples are transparent. */
export function colorizeDepth(meters: Float32Array, nearM = 0.3, farM = 5, out = new Uint8Array(meters.length * 4)): Uint8Array {
  if (out.length !== meters.length * 4) throw new Error('Colorized depth buffer has the wrong size');
  for (let i = 0; i < meters.length; i++) {
    const offset = i * 4;
    if (!meters[i]) { out[offset + 3] = 0; continue; }
    const t = Math.min(1, Math.max(0, (meters[i] - nearM) / (farM - nearM)));
    out[offset] = Math.round(255 * (1 - t));
    out[offset + 1] = Math.round(255 * (1 - Math.abs(2 * t - 1)));
    out[offset + 2] = Math.round(255 * t);
    out[offset + 3] = 255;
  }
  return out;
}
