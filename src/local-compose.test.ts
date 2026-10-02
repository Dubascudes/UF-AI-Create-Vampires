import { describe, expect, it } from 'vitest';
import { bytesToUnit, liveAlphaFromPersonMask, maskCoverage, shouldRequestFill } from './local-compose';

describe('liveAlphaFromPersonMask', () => {
  it('hides the live camera where a person is masked', () => {
    const out = liveAlphaFromPersonMask(new Uint8Array([0, 255, 128]), new Uint8ClampedArray(12));
    expect([out[3], out[7], out[11]]).toEqual([255, 0, 127]);
  });

  it('rejects mismatched buffers', () => {
    expect(() => liveAlphaFromPersonMask(new Uint8Array(2), new Uint8ClampedArray(4))).toThrow();
  });
});

describe('maskCoverage', () => {
  it('counts pixels at or above the threshold', () => {
    expect(maskCoverage(new Uint8Array([0, 128, 255, 10]))).toBe(0.5);
    expect(maskCoverage(new Uint8Array())).toBe(0);
  });
});

describe('shouldRequestFill', () => {
  const ready = { pending: false, lastRequestAt: 0, knownFraction: 0.5, missingFraction: 0.1 };
  it('requests when conditions hold', () => expect(shouldRequestFill(ready, 5000)).toBe(true));
  it('waits for a pending request, too little room, nothing missing, or the interval', () => {
    expect(shouldRequestFill({ ...ready, pending: true }, 5000)).toBe(false);
    expect(shouldRequestFill({ ...ready, knownFraction: 0.1 }, 5000)).toBe(false);
    expect(shouldRequestFill({ ...ready, missingFraction: 0 }, 5000)).toBe(false);
    expect(shouldRequestFill(ready, 2000)).toBe(false);
  });
});

describe('bytesToUnit', () => {
  it('scales bytes to 0–1', () => {
    const values = bytesToUnit(new Uint8Array([0, 51, 255]));
    expect(values[0]).toBe(0);
    expect(values[1]).toBeCloseTo(0.2, 6);
    expect(values[2]).toBe(1);
  });
});
