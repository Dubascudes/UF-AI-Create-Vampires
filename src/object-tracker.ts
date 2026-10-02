interface ColorSample { x: number; y: number; red: number; green: number; blue: number }

export class ObjectTracker {
  offsetX = 0;
  offsetY = 0;

  private constructor(
    readonly width: number,
    readonly height: number,
    readonly mask: Float32Array,
    private readonly samples: ColorSample[],
  ) {}

  static create(frame: Uint8ClampedArray, mask: Float32Array, width: number, height: number): ObjectTracker | null {
    if (mask.length !== width * height || frame.length !== width * height * 4) return null;
    let area = 0;
    const samples: ColorSample[] = [];
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const index = y * width + x;
        if (mask[index] < 0.65) continue;
        area++;
        if (x % 4 || y % 4) continue;
        const pixel = index * 4;
        samples.push({ x, y, red: frame[pixel], green: frame[pixel + 1], blue: frame[pixel + 2] });
      }
    }
    if (area < 12 || area > width * height * 0.45) return null;
    if (samples.length < 4) {
      for (let i = 0; i < mask.length; i++) {
        if (mask[i] < 0.65) continue;
        const pixel = i * 4;
        samples.push({ x: i % width, y: Math.floor(i / width), red: frame[pixel], green: frame[pixel + 1], blue: frame[pixel + 2] });
      }
    }
    const stride = Math.max(1, Math.ceil(samples.length / 160));
    return new ObjectTracker(width, height, mask.slice(), samples.filter((_, i) => i % stride === 0));
  }

  private score(frame: Uint8ClampedArray, dx: number, dy: number): number {
    let error = 0;
    for (const sample of this.samples) {
      const x = sample.x + dx;
      const y = sample.y + dy;
      if (x < 0 || x >= this.width || y < 0 || y >= this.height) return Infinity;
      const pixel = (y * this.width + x) * 4;
      error += Math.abs(frame[pixel] - sample.red)
        + Math.abs(frame[pixel + 1] - sample.green)
        + Math.abs(frame[pixel + 2] - sample.blue);
    }
    return error / (this.samples.length * 3)
      + 0.025 * ((dx - this.offsetX) ** 2 + (dy - this.offsetY) ** 2);
  }

  update(frame: Uint8ClampedArray): boolean {
    if (frame.length !== this.width * this.height * 4) return false;
    let bestX = this.offsetX;
    let bestY = this.offsetY;
    let best = this.score(frame, bestX, bestY);
    for (let dy = this.offsetY - 48; dy <= this.offsetY + 48; dy += 4) {
      for (let dx = this.offsetX - 48; dx <= this.offsetX + 48; dx += 4) {
        const score = this.score(frame, dx, dy);
        if (score < best) { best = score; bestX = dx; bestY = dy; }
      }
    }
    const coarseX = bestX;
    const coarseY = bestY;
    for (let dy = coarseY - 4; dy <= coarseY + 4; dy++) {
      for (let dx = coarseX - 4; dx <= coarseX + 4; dx++) {
        const score = this.score(frame, dx, dy);
        if (score < best) { best = score; bestX = dx; bestY = dy; }
      }
    }
    if (best > 55) return false;
    this.offsetX = bestX;
    this.offsetY = bestY;
    return true;
  }

  projectMask(width: number, height: number): Float32Array {
    const projected = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
      const sourceY = Math.floor((y + 0.5) * this.height / height) - this.offsetY;
      if (sourceY < 0 || sourceY >= this.height) continue;
      for (let x = 0; x < width; x++) {
        const sourceX = Math.floor((x + 0.5) * this.width / width) - this.offsetX;
        if (sourceX < 0 || sourceX >= this.width) continue;
        projected[y * width + x] = this.mask[sourceY * this.width + sourceX];
      }
    }
    return projected;
  }
}
