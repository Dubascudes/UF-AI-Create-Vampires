import { describe, expect, it } from 'vitest';
import { composeHybrid, conservativePersonMask, erasePersonForModel } from './hybrid';

function pixels(...reds: number[]): Uint8ClampedArray {
  return Uint8ClampedArray.from(reds.flatMap(red => [red, 0, 0, 255]));
}

function reds(image: Uint8ClampedArray): number[] {
  return Array.from({ length: image.length / 4 }, (_, index) => image[index * 4]);
}

describe('hybrid person removal', () => {
  it('covers face edges conservatively and holds a brief segmentation dropout', () => {
    const confidence = new Float32Array(11 * 11);
    confidence[5 * 11 + 5] = 0.9;
    const first = conservativePersonMask(confidence, 11, 11);
    expect(first.mask[5 * 11 + 5]).toBe(255);
    expect(first.mask[5 * 11 + 8]).toBe(255);
    expect(first.mask[0]).toBe(0);

    const dropped = conservativePersonMask(new Float32Array(11 * 11), 11, 11, first.temporal);
    expect(dropped.mask[5 * 11 + 5]).toBe(255);
  });

  it('sends a person-erased model input while keeping unmasked and selected-object pixels', () => {
    const result = erasePersonForModel(
      pixels(10, 20, 30), pixels(100, 110, 120),
      Uint8Array.from([0, 255, 255]), Uint8Array.from([0, 0, 255]),
    );
    expect(reds(result)).toEqual([10, 110, 30]);
  });

  it('uses persistent room pixels first, safe generated fill second, and never shows a generated person', () => {
    const raw = pixels(10, 20, 30, 40, 50);
    const room = pixels(100, 110, 120, 130, 140);
    const generated = pixels(200, 210, 220, 230, 240);
    const result = composeHybrid(raw, room, generated, {
      person: Uint8Array.from([0, 255, 255, 255, 255]),
      known: Uint8Array.from([0, 255, 0, 0, 0]),
      generatedPerson: Uint8Array.from([0, 0, 0, 255, 0]),
      object: Uint8Array.from([0, 0, 0, 0, 255]),
    });
    expect(reds(result)).toEqual([10, 110, 220, 130, 50]);
    expect(reds(composeHybrid(raw, room, generated, {
      person: Uint8Array.from([0, 255, 255, 255, 255]),
      known: Uint8Array.from([0, 255, 0, 0, 0]),
      generatedPerson: null,
      object: Uint8Array.from([0, 0, 0, 0, 255]),
    }))).toEqual([10, 110, 120, 130, 50]);
  });
});
