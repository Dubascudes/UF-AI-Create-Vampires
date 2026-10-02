import { describe, expect, it } from 'vitest';
import { colorizeDepth, depthStats, depthToMeters } from './xr-depth';

describe('depthToMeters', () => {
  it('scales 16-bit luminance-alpha values and zeroes invalid samples', () => {
    const raw = new Uint16Array([0, 1000, 2500, 500]);
    expect(Array.from(depthToMeters(raw.buffer, 'luminance-alpha', 0.001, 2, 2))).toEqual([0, 1, 2.5, 0.5]);
  });

  it('reads float32 depth', () => {
    const raw = new Float32Array([1.5, Number.NaN]);
    expect(Array.from(depthToMeters(raw.buffer, 'float32', 1, 2, 1))).toEqual([1.5, 0]);
  });

  it('rejects a buffer smaller than its dimensions', () => {
    expect(() => depthToMeters(new ArrayBuffer(6), 'unsigned-short', 1, 2, 2)).toThrow();
  });
});

describe('depthStats', () => {
  it('ignores invalid samples', () => {
    expect(depthStats(new Float32Array([0, 3, 1, 2]))).toEqual({ validFraction: 0.75, min: 1, median: 2, max: 3 });
  });

  it('handles empty depth', () => {
    expect(depthStats(new Float32Array(4)).validFraction).toBe(0);
  });
});

describe('colorizeDepth', () => {
  it('makes invalid depth transparent and near depth red', () => {
    const colors = colorizeDepth(new Float32Array([0, 0.3]), 0.3, 5);
    expect(colors[3]).toBe(0);
    expect(Array.from(colors.subarray(4))).toEqual([255, 0, 0, 255]);
  });
});
