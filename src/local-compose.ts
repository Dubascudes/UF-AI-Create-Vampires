/**
 * Write the live-camera opacity for each pixel into the alpha channel of an RGBA buffer:
 * 255 where no person is detected, 0 where the (dilated) person mask is set.
 */
export function liveAlphaFromPersonMask(person: Uint8Array, out: Uint8ClampedArray): Uint8ClampedArray {
  if (out.length !== person.length * 4) throw new Error('Alpha buffer does not match the person mask');
  for (let i = 0; i < person.length; i++) {
    const offset = i * 4;
    out[offset] = out[offset + 1] = out[offset + 2] = 255;
    out[offset + 3] = 255 - person[i];
  }
  return out;
}

export function maskCoverage(mask: Uint8Array, threshold = 128): number {
  if (!mask.length) return 0;
  let count = 0;
  for (const value of mask) if (value >= threshold) count++;
  return count / mask.length;
}

export interface FillState { pending: boolean; lastRequestAt: number; knownFraction: number; missingFraction: number }

/** Ask the inpainter only when enough room is known to condition on, something is still missing, and not too often. */
export function shouldRequestFill(state: FillState, now: number, minIntervalMs = 3000): boolean {
  return !state.pending && state.knownFraction >= 0.15 && state.missingFraction >= 0.005 &&
    now - state.lastRequestAt >= minIntervalMs;
}

/** Bytes 0–255 to floats 0–1; a plain loop is several times faster than Float32Array.from with a callback. */
export function bytesToUnit(bytes: Uint8Array, out = new Float32Array(bytes.length)): Float32Array {
  for (let i = 0; i < bytes.length; i++) out[i] = bytes[i] / 255;
  return out;
}
