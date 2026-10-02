import { applyColorTransform, estimateColorTransform, type ColorTransform } from './exposure';

export interface BackgroundResult {
  background: Uint8ClampedArray;
  mask: Float32Array;
  missingMask: Uint8Array;
  knownMask: Uint8Array;
  missingFraction: number;
  knownFraction: number;
}

function expandPerson(person: Float32Array, width: number, height: number): Float32Array {
  const expanded = person.slice();
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const value = person[y * width + x];
      if (value < 0.35) continue;
      for (let dy = -2; dy <= 2; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        for (let dx = -2; dx <= 2; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= width) continue;
          const distance = Math.max(Math.abs(dx), Math.abs(dy));
          const candidate = value * (distance === 2 ? 0.85 : 1);
          const index = ny * width + nx;
          if (candidate > expanded[index]) expanded[index] = candidate;
        }
      }
    }
  }
  return expanded;
}

export class BackgroundModel {
  private readonly reference: Uint8ClampedArray;
  private readonly known: Uint8Array;
  private readonly synthesized: Uint8Array;
  private readonly candidate: Uint8ClampedArray;
  private readonly streak: Uint8Array;
  private previousPersonMask = new Float32Array(0);
  private count = 0;
  private transform: ColorTransform = { gain: [1, 1, 1], offset: [0, 0, 0] };

  constructor(readonly width: number, readonly height: number) {
    this.reference = new Uint8ClampedArray(width * height * 4);
    this.known = new Uint8Array(width * height);
    this.synthesized = new Uint8Array(width * height);
    this.candidate = new Uint8ClampedArray(width * height * 4);
    this.streak = new Uint8Array(width * height);
  }

  get knownFraction(): number { return this.count / this.known.length; }

  reset(): void {
    this.reference.fill(0);
    this.known.fill(0);
    this.synthesized.fill(0);
    this.candidate.fill(0);
    this.streak.fill(0);
    this.previousPersonMask = new Float32Array(0);
    this.count = 0;
    this.transform = { gain: [1, 1, 1], offset: [0, 0, 0] };
  }

  installGenerated(image: Uint8ClampedArray, missingMask: Uint8Array): void {
    if (image.length !== this.reference.length || missingMask.length !== this.known.length) {
      throw new Error('Generated image and mask dimensions do not match');
    }
    for (let i = 0; i < this.known.length; i++) {
      if (!missingMask[i] || this.known[i] || this.synthesized[i]) continue;
      const offset = i * 4;
      for (let channel = 0; channel < 3; channel++) {
        this.reference[offset + channel] = (image[offset + channel] - this.transform.offset[channel]) / this.transform.gain[channel];
      }
      this.reference[offset + 3] = 255;
      this.synthesized[i] = 1;
    }
  }

  forget(object: Float32Array, maskWidth: number, maskHeight: number): void {
    if (object.length !== maskWidth * maskHeight) throw new Error('Object mask dimensions do not match');
    const remove = new Uint8Array(this.known.length);
    for (let y = 0; y < this.height; y++) {
      const maskY = Math.min(maskHeight - 1, Math.floor((y + 0.5) * maskHeight / this.height));
      for (let x = 0; x < this.width; x++) {
        const maskX = Math.min(maskWidth - 1, Math.floor((x + 0.5) * maskWidth / this.width));
        if (object[maskY * maskWidth + maskX] < 0.5) continue;
        for (let dy = -4; dy <= 4; dy++) {
          const ny = y + dy;
          if (ny < 0 || ny >= this.height) continue;
          for (let dx = -4; dx <= 4; dx++) {
            const nx = x + dx;
            if (nx >= 0 && nx < this.width) remove[ny * this.width + nx] = 1;
          }
        }
      }
    }
    for (let i = 0; i < remove.length; i++) {
      if (!remove[i]) continue;
      if (this.known[i]) this.count--;
      this.known[i] = 0;
      this.synthesized[i] = 0;
      this.streak[i] = 0;
    }
  }

  observe(frame: Uint8ClampedArray, person: Float32Array, maskWidth: number, maskHeight: number, preservedObject?: Float32Array): BackgroundResult {
    if (frame.length !== this.reference.length || person.length !== maskWidth * maskHeight) {
      throw new Error('Background frame and mask dimensions do not match');
    }
    if (preservedObject && preservedObject.length !== person.length) throw new Error('Object mask dimensions do not match');
    const spatialMask = expandPerson(person, maskWidth, maskHeight);
    const mask = spatialMask.slice();
    if (this.previousPersonMask.length === mask.length) {
      for (let i = 0; i < mask.length; i++) {
        mask[i] = Math.max(mask[i], this.previousPersonMask[i] * 0.8);
      }
    }
    this.previousPersonMask = spatialMask;
    const excluded = preservedObject
      ? mask.map((value, index) => Math.max(value, preservedObject[index]))
      : mask;
    const estimate = estimateColorTransform(this.reference, frame, excluded, this.width, this.height, maskWidth, maskHeight, this.known);
    if (estimate) this.transform = estimate;

    for (let y = 0; y < this.height; y++) {
      const maskY = Math.min(maskHeight - 1, Math.floor((y + 0.5) * maskHeight / this.height));
      for (let x = 0; x < this.width; x++) {
        const index = y * this.width + x;
        const maskX = Math.min(maskWidth - 1, Math.floor((x + 0.5) * maskWidth / this.width));
        if (excluded[maskY * maskWidth + maskX] >= 0.08) {
          if (!this.known[index]) this.streak[index] = 0;
          continue;
        }
        const offset = index * 4;
        if (!this.known[index]) {
          let changed = this.streak[index] === 0;
          for (let channel = 0; channel < 3; channel++) {
            const referenceValue = (frame[offset + channel] - this.transform.offset[channel]) / this.transform.gain[channel];
            if (Math.abs(referenceValue - this.candidate[offset + channel]) > 18) changed = true;
          }
          if (changed) this.streak[index] = 0;
          for (let channel = 0; channel < 3; channel++) {
            const referenceValue = (frame[offset + channel] - this.transform.offset[channel]) / this.transform.gain[channel];
            this.candidate[offset + channel] = this.streak[index]
              ? this.candidate[offset + channel] * 0.7 + referenceValue * 0.3
              : referenceValue;
          }
          this.streak[index]++;
          if (this.streak[index] < 5) continue;
          this.known[index] = 1;
          this.synthesized[index] = 0;
          this.count++;
          this.reference.set(this.candidate.subarray(offset, offset + 3), offset);
          this.reference[offset + 3] = 255;
          continue;
        }
        for (let channel = 0; channel < 3; channel++) {
          const referenceValue = (frame[offset + channel] - this.transform.offset[channel]) / this.transform.gain[channel];
          this.reference[offset + channel] = this.reference[offset + channel] * 0.97 + referenceValue * 0.03;
        }
        this.reference[offset + 3] = 255;
      }
    }

    const filled = this.fillUnknown();
    const background = new Uint8ClampedArray(filled.length);
    applyColorTransform(filled, this.transform, background);
    const missingMask = new Uint8Array(this.known.length);
    let missingCount = 0;
    for (let i = 0; i < missingMask.length; i++) {
      if (!this.known[i] && !this.synthesized[i]) { missingMask[i] = 255; missingCount++; }
    }
    const knownMask = this.known.map(value => value ? 255 : 0);
    return { background, mask, missingMask, knownMask, missingFraction: missingCount / missingMask.length, knownFraction: this.knownFraction };
  }

  private fillUnknown(): Uint8ClampedArray {
    const filled = this.reference.slice();
    if (this.count === this.known.length) return filled;
    const seen = this.known.slice();
    let seedCount = this.count;
    for (let i = 0; i < seen.length; i++) {
      if (this.synthesized[i] && !seen[i]) { seen[i] = 1; seedCount++; }
    }
    if (seedCount === 0) {
      for (let i = 0; i < filled.length; i += 4) {
        filled[i] = 25; filled[i + 1] = 35; filled[i + 2] = 30; filled[i + 3] = 255;
      }
      return filled;
    }
    const queue = new Int32Array(this.known.length);
    let tail = 0;
    for (let i = 0; i < seen.length; i++) if (seen[i]) queue[tail++] = i;
    let head = 0;
    while (head < tail) {
      const source = queue[head++];
      const x = source % this.width;
      const y = Math.floor(source / this.width);
      for (let direction = 0; direction < 4; direction++) {
        const target = direction === 0 ? (x > 0 ? source - 1 : -1)
          : direction === 1 ? (x < this.width - 1 ? source + 1 : -1)
          : direction === 2 ? (y > 0 ? source - this.width : -1)
          : (y < this.height - 1 ? source + this.width : -1);
        if (target < 0 || seen[target]) continue;
        seen[target] = 1;
        queue[tail++] = target;
        const from = source * 4;
        const to = target * 4;
        filled[to] = filled[from];
        filled[to + 1] = filled[from + 1];
        filled[to + 2] = filled[from + 2];
        filled[to + 3] = 255;
      }
    }
    return filled;
  }
}
