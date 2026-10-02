import { describe, expect, it } from 'vitest';
import { BackgroundModel } from './background';
import { liveOpacity } from './mask';

function frame(width: number, height: number, shift = 0): Uint8ClampedArray {
  const result = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      result[i] = 45 + x * 4 + shift;
      result[i + 1] = 60 + y * 3 + shift;
      result[i + 2] = 70 + x * 2 + y + shift;
      result[i + 3] = 255;
    }
  }
  return result;
}

describe('automatic background model', () => {
  it('fully removes the fringe just outside a confidently detected person', () => {
    const model = new BackgroundModel(20, 20);
    const person = new Float32Array(400);
    person[10 * 20 + 10] = 1;
    const result = model.observe(frame(20, 20), person, 20, 20);
    expect(liveOpacity(result.mask[10 * 20 + 12], 0.5)).toBe(0);
    expect(liveOpacity(result.mask[10 * 20 + 13], 0.5)).toBe(255);
  });

  it('holds a detected edge through one dropped mask and then releases it', () => {
    const model = new BackgroundModel(20, 20);
    const person = new Float32Array(400);
    person[10 * 20 + 10] = 1;
    model.observe(frame(20, 20), person, 20, 20);
    const empty = new Float32Array(400);
    let result = model.observe(frame(20, 20), empty, 20, 20);
    expect(liveOpacity(result.mask[10 * 20 + 10], 0.5)).toBe(0);
    for (let i = 0; i < 6; i++) result = model.observe(frame(20, 20), empty, 20, 20);
    expect(liveOpacity(result.mask[10 * 20 + 10], 0.5)).toBe(255);
  });

  it('learns the room as pixels become visible without a capture action', () => {
    const model = new BackgroundModel(16, 16);
    const hidden = new Float32Array(256);
    for (let y = 6; y <= 9; y++) for (let x = 6; x <= 9; x++) hidden[y * 16 + x] = 1;
    let first = model.observe(frame(16, 16), hidden, 16, 16);
    for (let i = 0; i < 5; i++) first = model.observe(frame(16, 16), hidden, 16, 16);
    expect(first.knownFraction).toBeGreaterThan(0);
    expect(first.knownFraction).toBeLessThan(1);
    expect(first.knownMask[7 * 16 + 7]).toBe(0);
    expect(first.knownMask[0]).toBe(255);
    let learned = first;
    for (let i = 0; i < 6; i++) learned = model.observe(frame(16, 16), new Float32Array(256), 16, 16);
    expect(learned.knownFraction).toBe(1);
    expect(learned.knownMask[7 * 16 + 7]).toBe(255);
  });

  it('adapts a previously learned hidden region to an exposure change', () => {
    const model = new BackgroundModel(16, 16);
    for (let i = 0; i < 6; i++) model.observe(frame(16, 16), new Float32Array(256), 16, 16);
    const person = new Float32Array(256);
    for (let y = 6; y <= 9; y++) for (let x = 6; x <= 9; x++) person[y * 16 + x] = 1;
    const brighter = frame(16, 16, 20);
    for (let y = 6; y <= 9; y++) for (let x = 6; x <= 9; x++) {
      const i = (y * 16 + x) * 4;
      brighter[i] = brighter[i + 1] = brighter[i + 2] = 245;
    }
    const result = model.observe(brighter, person, 16, 16);
    const center = (7 * 16 + 7) * 4;
    expect(result.background[center]).toBeCloseTo(frame(16, 16, 20)[center], 0);
    expect(result.background[center]).toBeLessThan(200);
  });

  it('resets learned pixels when the camera is moved', () => {
    const model = new BackgroundModel(16, 16);
    for (let i = 0; i < 6; i++) model.observe(frame(16, 16), new Float32Array(256), 16, 16);
    model.reset();
    expect(model.knownFraction).toBe(0);
  });

  it('does not memorize a person missed in one frame', () => {
    const model = new BackgroundModel(16, 16);
    const withPerson = frame(16, 16);
    const center = (7 * 16 + 7) * 4;
    withPerson[center] = withPerson[center + 1] = withPerson[center + 2] = 245;
    model.observe(withPerson, new Float32Array(256), 16, 16);
    const detected = new Float32Array(256);
    detected[7 * 16 + 7] = 1;
    for (let i = 0; i < 6; i++) model.observe(withPerson, detected, 16, 16);
    expect(model.knownFraction).toBeLessThan(1);
  });

  it('fills an unseen person region from nearby learned pixels', () => {
    const model = new BackgroundModel(16, 16);
    const person = new Float32Array(256);
    person[7 * 16 + 7] = 1;
    const live = frame(16, 16);
    const center = (7 * 16 + 7) * 4;
    live[center] = live[center + 1] = live[center + 2] = 245;
    let result = model.observe(live, person, 16, 16);
    for (let i = 0; i < 6; i++) result = model.observe(live, person, 16, 16);
    expect(result.background[center]).toBeGreaterThan(0);
    expect(result.background[center]).toBeLessThan(200);
  });

  it('does not memorize a selected floating object as room background', () => {
    const model = new BackgroundModel(16, 16);
    const object = new Float32Array(256);
    object[7 * 16 + 7] = 1;
    const live = frame(16, 16);
    for (let i = 0; i < 6; i++) model.observe(live, new Float32Array(256), 16, 16, object);
    expect(model.knownFraction).toBeLessThan(1);
  });

  it('forgets an object that was learned before selection', () => {
    const model = new BackgroundModel(24, 24);
    const live = frame(24, 24);
    const object = new Float32Array(24 * 24);
    for (let y = 10; y < 14; y++) for (let x = 10; x < 14; x++) {
      object[y * 24 + x] = 1;
      const i = (y * 24 + x) * 4;
      live[i] = 250;
    }
    for (let i = 0; i < 6; i++) model.observe(live, new Float32Array(24 * 24), 24, 24);
    expect(model.knownFraction).toBe(1);
    model.forget(object, 24, 24);
    const result = model.observe(live, new Float32Array(24 * 24), 24, 24, object);
    expect(result.knownFraction).toBeLessThan(1);
    expect(result.background[(11 * 24 + 11) * 4]).toBeLessThan(200);
  });

  it('uses generated pixels only until the real room is revealed', () => {
    const model = new BackgroundModel(24, 24);
    const person = new Float32Array(24 * 24);
    for (let y = 9; y < 15; y++) for (let x = 9; x < 15; x++) person[y * 24 + x] = 1;
    let before = model.observe(frame(24, 24), person, 24, 24);
    for (let i = 0; i < 6; i++) before = model.observe(frame(24, 24), person, 24, 24);
    const center = 12 * 24 + 12;
    expect(before.missingMask[center]).toBe(255);
    const generated = new Uint8ClampedArray(24 * 24 * 4);
    for (let i = 0; i < generated.length; i += 4) {
      generated[i] = 185; generated[i + 1] = 160; generated[i + 2] = 140; generated[i + 3] = 255;
    }
    model.installGenerated(generated, before.missingMask);
    const filled = model.observe(frame(24, 24), person, 24, 24);
    expect(filled.background[center * 4]).toBeGreaterThan(175);
    expect(filled.missingMask[center]).toBe(0);
    expect(filled.knownFraction).toBeLessThan(1);
    let revealed = filled;
    for (let i = 0; i < 7; i++) revealed = model.observe(frame(24, 24), new Float32Array(24 * 24), 24, 24);
    expect(revealed.knownFraction).toBe(1);
    expect(revealed.background[center * 4]).toBeLessThan(150);
  });
});
