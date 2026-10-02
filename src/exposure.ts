export interface ColorTransform {
  gain: [number, number, number];
  offset: [number, number, number];
}

const clamp = (value: number, lower: number, upper: number): number =>
  Math.max(lower, Math.min(upper, value));

function median(values: number[]): number {
  values.sort((a, b) => a - b);
  return values[Math.floor(values.length / 2)];
}

/** Estimate a robust per-channel affine match using only uncovered background. */
export function estimateColorTransform(
  plate: Uint8ClampedArray,
  live: Uint8ClampedArray,
  person: Float32Array,
  width: number,
  height: number,
  maskWidth: number,
  maskHeight: number,
  known?: Uint8Array,
): ColorTransform | null {
  const samples: number[][] = [];
  const stride = Math.max(1, Math.floor(Math.sqrt(width * height / 256)));
  for (let y = 0; y < height; y += stride) {
    const maskY = Math.min(maskHeight - 1, Math.floor((y + 0.5) * maskHeight / height));
    for (let x = 0; x < width; x += stride) {
      const maskX = Math.min(maskWidth - 1, Math.floor((x + 0.5) * maskWidth / width));
      if (person[maskY * maskWidth + maskX] > 0.1) continue;
      if (known && !known[y * width + x]) continue;
      const i = (y * width + x) * 4;
      samples.push([plate[i], plate[i + 1], plate[i + 2], live[i], live[i + 1], live[i + 2]]);
    }
  }
  if (samples.length < 12) return null;
  const step = Math.ceil(samples.length / 96);
  const selected = samples.filter((_, i) => i % step === 0);
  const gain: [number, number, number] = [1, 1, 1];
  const offset: [number, number, number] = [0, 0, 0];

  for (let channel = 0; channel < 3; channel++) {
    const slopes: number[] = [];
    for (let i = 0; i < selected.length; i++) {
      for (let j = i + 1; j < selected.length; j++) {
        const difference = selected[j][channel] - selected[i][channel];
        if (Math.abs(difference) < 12) continue;
        slopes.push((selected[j][channel + 3] - selected[i][channel + 3]) / difference);
      }
    }
    const slope = slopes.length ? clamp(median(slopes), 0.6, 1.6) : 1;
    const intercept = median(selected.map(sample => sample[channel + 3] - sample[channel] * slope));
    gain[channel] = slope;
    offset[channel] = clamp(intercept, -64, 64);
  }
  return { gain, offset };
}

export function applyColorTransform(
  source: Uint8ClampedArray,
  transform: ColorTransform,
  target: Uint8ClampedArray,
): void {
  for (let i = 0; i < source.length; i += 4) {
    for (let channel = 0; channel < 3; channel++) {
      target[i + channel] = source[i + channel] * transform.gain[channel] + transform.offset[channel];
    }
    target[i + 3] = source[i + 3];
  }
}
