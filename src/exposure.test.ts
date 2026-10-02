import { describe, expect, it } from 'vitest';
import { applyColorTransform, estimateColorTransform } from './exposure';

function scene(width: number, height: number) {
  const plate = new Uint8ClampedArray(width * height * 4);
  const live = new Uint8ClampedArray(width * height * 4);
  const person = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = y * width + x;
      const offset = index * 4;
      const values = [35 + x * 12 + y * 3, 45 + x * 8 + y * 5, 55 + x * 7 + y * 4];
      const gains = [1.2, 0.9, 1.1];
      const shifts = [12, 18, -8];
      const hidden = x >= 3 && x <= 5 && y >= 2 && y <= 5;
      person[index] = hidden ? 0.95 : 0.02;
      for (let channel = 0; channel < 3; channel++) {
        plate[offset + channel] = values[channel];
        live[offset + channel] = hidden ? 240 : Math.round(values[channel] * gains[channel] + shifts[channel]);
      }
      plate[offset + 3] = 255;
      live[offset + 3] = 255;
    }
  }
  return { plate, live, person };
}

describe('background exposure matching', () => {
  it('recovers per-channel camera exposure changes from visible background', () => {
    const { plate, live, person } = scene(8, 8);
    const transform = estimateColorTransform(plate, live, person, 8, 8, 8, 8);
    expect(transform).not.toBeNull();
    expect(transform!.gain[0]).toBeCloseTo(1.2, 1);
    expect(transform!.gain[1]).toBeCloseTo(0.9, 1);
    expect(transform!.gain[2]).toBeCloseTo(1.1, 1);
    expect(transform!.offset[0]).toBeCloseTo(12, 0);
    expect(transform!.offset[1]).toBeCloseTo(18, 0);
    expect(transform!.offset[2]).toBeCloseTo(-8, 0);
  });

  it('ignores a changed object outside the person mask', () => {
    const { plate, live, person } = scene(8, 8);
    for (const index of [0, 7, 56, 63]) {
      for (let channel = 0; channel < 3; channel++) live[index * 4 + channel] = 0;
    }
    const transform = estimateColorTransform(plate, live, person, 8, 8, 8, 8);
    expect(transform).not.toBeNull();
    expect(transform!.gain[0]).toBeCloseTo(1.2, 1);
    expect(transform!.offset[0]).toBeCloseTo(12, 0);
  });

  it('does not infer exposure when no background is visible', () => {
    const { plate, live } = scene(8, 8);
    const person = new Float32Array(64).fill(1);
    expect(estimateColorTransform(plate, live, person, 8, 8, 8, 8)).toBeNull();
  });

  it('maps the saved plate to the live camera brightness without changing alpha', () => {
    const source = new Uint8ClampedArray([100, 80, 120, 255]);
    const target = new Uint8ClampedArray(4);
    applyColorTransform(source, { gain: [1.2, 0.9, 1.1], offset: [12, 18, -8] }, target);
    expect(Array.from(target)).toEqual([132, 90, 124, 255]);
  });
});
