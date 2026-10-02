import { describe, expect, it } from 'vitest';
import { ObjectTracker } from './object-tracker';

function image(x: number, y: number): Uint8ClampedArray {
  const data = new Uint8ClampedArray(64 * 48 * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = data[i + 1] = data[i + 2] = 70;
    data[i + 3] = 255;
  }
  for (let row = y; row < y + 8; row++) {
    for (let column = x; column < x + 10; column++) {
      const i = (row * 64 + column) * 4;
      data[i] = 215;
      data[i + 1] = 30 + column - x;
      data[i + 2] = 40 + row - y;
    }
  }
  return data;
}

describe('selected object tracking', () => {
  it('moves a preserved object mask with a translated object', () => {
    const mask = new Float32Array(64 * 48);
    for (let y = 15; y < 23; y++) for (let x = 18; x < 28; x++) mask[y * 64 + x] = 1;
    const tracker = ObjectTracker.create(image(18, 15), mask, 64, 48);
    expect(tracker).not.toBeNull();
    expect(tracker!.update(image(23, 18))).toBe(true);
    expect(tracker!.offsetX).toBe(5);
    expect(tracker!.offsetY).toBe(3);
    const moved = tracker!.projectMask(64, 48);
    expect(moved[20 * 64 + 25]).toBe(1);
    expect(moved[15 * 64 + 18]).toBe(0);
  });

  it('rejects a selection covering most of the camera view', () => {
    expect(ObjectTracker.create(image(18, 15), new Float32Array(64 * 48).fill(1), 64, 48)).toBeNull();
  });
});
